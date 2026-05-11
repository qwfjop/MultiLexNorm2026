from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from dataset_io import load_dataset_auto
from utils import evaluate, gold_or_raw, zip_files_flat

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _load_dataset(dataset_name: str, use_auth_token: bool = False):
    return load_dataset_auto(dataset_name, use_auth_token=use_auth_token)


def _select_device():
    import torch

    if torch.cuda.is_available():
        print("CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("MPS")
        return torch.device("mps")
    print("CPU")
    return torch.device("cpu")


def _as_token_list(value) -> list[str] | None:
    if isinstance(value, str):
        return None
    try:
        tokens = list(value)
    except TypeError:
        return None
    if not all(isinstance(token, str) for token in tokens):
        return None
    return tokens


def format_prompt(raw_tokens: Sequence[str], lang: str, target_index: int) -> str:
    target_token = raw_tokens[target_index]
    return (
        "Normalize only the target token using the sentence context.\n"
        "Return only the normalized token. Do not explain.\n\n"
        f"Language: {lang}\n"
        f"Sentence tokens: {json.dumps(list(raw_tokens), ensure_ascii=False)}\n"
        f"Target index: {target_index}\n"
        f"Target token: {json.dumps(target_token, ensure_ascii=False)}\n"
        "Normalized token:"
    )


def iter_context_token_examples(records: Iterable[Mapping[str, object]]):
    for record in records:
        lang = str(record["lang"])
        raw_tokens = _as_token_list(record["raw"])
        norm_tokens = _as_token_list(record["norm"])
        if raw_tokens is None or norm_tokens is None:
            continue

        for target_index, (raw_token, norm_token) in enumerate(zip(raw_tokens, norm_tokens)):
            yield {
                "prompt": format_prompt(raw_tokens, lang, target_index),
                "target": gold_or_raw(raw_token, norm_token),
            }


class ContextTokenNormalizationDataset:
    def __init__(self, examples, tokenizer, max_length: int) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        prompt_ids = self.tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        answer_text = " " + example["target"] + self.tokenizer.eos_token
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]

        if len(answer_ids) >= self.max_length:
            answer_ids = answer_ids[: self.max_length]
            prompt_ids = []
        else:
            prompt_ids = prompt_ids[-(self.max_length - len(answer_ids)) :]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class CausalTokenCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Sequence[int]]]) -> dict:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            pad_length = max_length - len(feature["input_ids"])
            input_ids.append(list(feature["input_ids"]) + [self.pad_token_id] * pad_length)
            attention_mask.append(list(feature["attention_mask"]) + [0] * pad_length)
            labels.append(list(feature["labels"]) + [-100] * pad_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class GemmaTrainConfig:
    model_name: str = "google/gemma-3-270m-it"
    output_dir: str = "models/gemma"
    max_length: int = 256
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    epochs: int = 1
    max_train_examples: int | None = None
    log_every_steps: int = 25
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    save_every_steps: int = 0


class GemmaNormalizer:
    def __init__(
        self,
        model_name: str = "google/gemma-3-270m-it",
        device=None,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or _select_device()
        self.adapter_dir = os.path.isdir(model_name) and os.path.exists(
            os.path.join(model_name, "adapter_config.json")
        )

        if self.adapter_dir:
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(model_name)
            base_model_name = peft_config.base_model_name_or_path
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if self.device.type in {"cuda", "mps"} else torch.float32
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
            self.model = PeftModel.from_pretrained(base_model, model_name, is_trainable=use_lora)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if self.device.type in {"cuda", "mps"} else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

            if use_lora:
                from peft import LoraConfig, get_peft_model

                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                )
                self.model = get_peft_model(self.model, lora_config)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        self.model.to(self.device)

        if use_lora and hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

    def fit(self, records: Iterable[Mapping[str, object]], config: GemmaTrainConfig) -> "GemmaNormalizer":
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader

        examples = list(iter_context_token_examples(records))
        if config.max_train_examples is not None:
            examples = examples[: config.max_train_examples]
        if not examples:
            raise ValueError("No training examples were produced from the dataset.")

        dataset = ContextTokenNormalizationDataset(
            examples,
            tokenizer=self.tokenizer,
            max_length=config.max_length,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=CausalTokenCollator(self.tokenizer.pad_token_id),
        )

        print(
            "training "
            f"examples={len(dataset)} batches_per_epoch={len(loader)} "
            f"batch_size={config.batch_size} grad_accum={config.gradient_accumulation_steps} "
            f"device={self.device} lora={config.use_lora}",
            flush=True,
        )

        self.model.train()
        optimizer = AdamW((p for p in self.model.parameters() if p.requires_grad), lr=config.learning_rate)
        optimizer.zero_grad(set_to_none=True)

        step_count = 0
        for epoch in range(config.epochs):
            running_loss = 0.0
            for batch_index, batch in enumerate(loader, start=1):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss / config.gradient_accumulation_steps
                loss.backward()
                batch_loss = float(loss.detach().cpu()) * config.gradient_accumulation_steps
                running_loss += batch_loss

                if batch_index % config.gradient_accumulation_steps == 0 or batch_index == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step_count += 1

                    if config.save_every_steps > 0 and step_count % config.save_every_steps == 0:
                        checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-step-{step_count}")
                        self.save(checkpoint_dir)
                        print(f"saved checkpoint={checkpoint_dir}", flush=True)

                if config.log_every_steps > 0 and (
                    batch_index == 1
                    or batch_index % config.log_every_steps == 0
                    or batch_index == len(loader)
                ):
                    avg_loss = running_loss / batch_index
                    print(
                        f"epoch={epoch + 1}/{config.epochs} "
                        f"batch={batch_index}/{len(loader)} "
                        f"optimizer_steps={step_count} "
                        f"batch_loss={batch_loss:.4f} avg_loss={avg_loss:.4f}",
                        flush=True,
                    )

            avg_loss = running_loss / max(len(loader), 1)
            print(f"epoch={epoch + 1} avg_loss={avg_loss:.4f} optimizer_steps={step_count}", flush=True)

        self.save(config.output_dir)
        return self

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_path: str, device=None) -> "GemmaNormalizer":
        return cls(model_path, device=device, use_lora=False)

    def predict_token_items(
        self,
        items: Sequence[tuple[list[str], str, int]],
        batch_size: int = 4,
        max_length: int = 256,
        max_new_tokens: int = 16,
        log_every_batches: int = 25,
    ) -> list[str]:
        import torch

        self.model.eval()
        predictions: list[str] = []
        total_batches = (len(items) + batch_size - 1) // batch_size
        print(
            f"predicting tokens={len(items)} batch_size={batch_size} batches={total_batches}",
            flush=True,
        )

        with torch.inference_mode():
            for batch_number, start in enumerate(range(0, len(items), batch_size), start=1):
                chunk = list(items[start : start + batch_size])
                prompts = [format_prompt(raw_tokens, lang, target_index) for raw_tokens, lang, target_index in chunk]
                encoded = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                input_lengths = encoded["attention_mask"].sum(dim=1).tolist()
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                for sequence, input_length in zip(generated, input_lengths):
                    new_token_ids = sequence[int(input_length) :]
                    decoded = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
                    predictions.append(clean_prediction(decoded))

                if log_every_batches > 0 and (
                    batch_number == 1
                    or batch_number % log_every_batches == 0
                    or batch_number == total_batches
                ):
                    print(f"predict batch={batch_number}/{total_batches}", flush=True)

        return predictions

    def predict_dataframe(
        self,
        data: pd.DataFrame,
        batch_size: int = 4,
        max_length: int = 256,
        max_new_tokens: int = 16,
    ) -> pd.DataFrame:
        out = data.copy()
        token_items: list[tuple[list[str], str, int]] = []
        row_lengths: list[int] = []

        for row in out.itertuples(index=False):
            raw_tokens = list(row.raw)
            row_lengths.append(len(raw_tokens))
            token_items.extend((raw_tokens, row.lang, index) for index in range(len(raw_tokens)))

        flat_predictions = self.predict_token_items(
            token_items,
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )

        preds: list[list[str]] = []
        cursor = 0
        for row_length in row_lengths:
            preds.append(flat_predictions[cursor : cursor + row_length])
            cursor += row_length

        out["pred"] = preds
        return out


def clean_prediction(text: str) -> str:
    text = text.strip()
    if "\n" in text:
        text = text.splitlines()[0].strip()
    if text.startswith("Normalized token:"):
        text = text[len("Normalized token:") :].strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return text.strip()


def _records_from_dataset(dataset) -> list[dict]:
    return list(dataset)


def train_from_hf_dataset(
    dataset_name: str,
    config: GemmaTrainConfig,
    use_auth_token: bool = False,
    cli_args: Sequence[str] | None = None,
) -> GemmaNormalizer:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    model = GemmaNormalizer(
        config.model_name,
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    model.fit(_records_from_dataset(data["train"]), config)
    write_training_metadata(
        config.output_dir,
        dataset_name=dataset_name,
        use_auth_token=use_auth_token,
        config=config,
        cli_args=cli_args,
    )
    return model


def write_training_metadata(
    output_dir: str,
    dataset_name: str,
    use_auth_token: bool,
    config: GemmaTrainConfig,
    cli_args: Sequence[str] | None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    command = " ".join(shlex.quote(part) for part in ([sys.argv[0]] + list(cli_args or [])))
    metadata = {
        "dataset": dataset_name,
        "use_auth_token": use_auth_token,
        "command": command,
        "config": asdict(config),
    }

    json_path = os.path.join(output_dir, "training_args.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(output_dir, "training_args.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"use_auth_token: {use_auth_token}\n")
        f.write(f"command: {command}\n\n")
        for key, value in asdict(config).items():
            f.write(f"{key}: {value}\n")

    print(f"Saved training metadata to {json_path} and {txt_path}", flush=True)


def evaluate_validation(
    dataset_name: str,
    model_path: str,
    use_auth_token: bool = False,
    batch_size: int = 4,
    max_length: int = 256,
    max_new_tokens: int = 16,
    max_eval_examples: int | None = None,
) -> tuple[float, float, float]:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    model = GemmaNormalizer.from_pretrained(model_path)
    validation = data["validation"]
    if max_eval_examples is not None:
        validation = validation.select(range(min(max_eval_examples, len(validation))))
    out = model.predict_dataframe(
        validation.to_pandas(),
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )
    return evaluate(
        raw=out["raw"].tolist(),
        gold=out["norm"].tolist(),
        pred=out["pred"].tolist(),
        info=True,
    )


def inspect_validation_predictions(
    dataset_name: str,
    model_path: str,
    use_auth_token: bool = False,
    batch_size: int = 4,
    limit: int = 10,
    langs: set[str] | None = None,
    max_length: int = 256,
    max_new_tokens: int = 16,
) -> None:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    validation = data["validation"]
    if langs is not None:
        validation = validation.filter(lambda row: row["lang"] in langs)

    model = GemmaNormalizer.from_pretrained(model_path)
    out = model.predict_dataframe(
        validation.select(range(min(limit, len(validation)))).to_pandas(),
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )

    for index, row in enumerate(out.itertuples(index=False), start=1):
        print(f"\n[{index}] lang={row.lang}")
        print("raw :", row.raw)
        print("gold:", [gold_or_raw(raw, gold) for raw, gold in zip(row.raw, row.norm)])
        print("pred:", row.pred)
        changed = [
            f"{raw} -> pred:{prediction} / gold:{gold_or_raw(raw, gold)}"
            for raw, gold, prediction in zip(row.raw, row.norm, row.pred)
            if prediction != gold_or_raw(raw, gold)
        ]
        print("diff:", "; ".join(changed) if changed else "all correct")


def create_submission(
    dataset_name: str,
    model_path: str,
    output_dir: str,
    use_auth_token: bool = False,
    batch_size: int = 4,
    max_length: int = 256,
    max_new_tokens: int = 16,
    zip_output: bool = True,
) -> str:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    model = GemmaNormalizer.from_pretrained(model_path)
    out = model.predict_dataframe(
        data["test"].to_pandas(),
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )

    os.makedirs(output_dir, exist_ok=True)
    prediction_path = os.path.join(output_dir, "predictions.json")
    out.to_json(prediction_path, orient="records")

    if zip_output:
        zip_files_flat(output_dir, f"{output_dir}.zip")

    return prediction_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma for context-aware MultiLexNorm normalization.")
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--model-name", default="google/gemma-3-270m-it")
    parser.add_argument("--model-dir", default="models/gemma")
    parser.add_argument("--output-dir", default="outputs/submission_gemma")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--no-lora", action="store_true", help="Train all Gemma weights. This is not recommended on T4.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--use-auth-token", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--inspect-limit", type=int, default=10)
    parser.add_argument("--inspect-langs", help="Comma-separated language filter, e.g. en,ko")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    config = GemmaTrainConfig(
        model_name=args.model_name,
        output_dir=args.model_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_train_examples=args.max_train_examples,
        log_every_steps=args.log_every_steps,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        save_every_steps=args.save_every_steps,
    )

    if args.train:
        train_from_hf_dataset(
            args.dataset,
            config,
            use_auth_token=args.use_auth_token,
            cli_args=sys.argv[1:],
        )
    if args.eval_only:
        evaluate_validation(
            args.dataset,
            args.model_dir,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            max_eval_examples=args.max_eval_examples,
        )
    if args.inspect:
        inspect_langs = None
        if args.inspect_langs:
            inspect_langs = {lang.strip() for lang in args.inspect_langs.split(",") if lang.strip()}
        inspect_validation_predictions(
            args.dataset,
            args.model_dir,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
            limit=args.inspect_limit,
            langs=inspect_langs,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
    if args.predict_test:
        create_submission(
            args.dataset,
            args.model_dir,
            args.output_dir,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            zip_output=not args.no_zip,
        )
    if not (args.train or args.eval_only or args.inspect or args.predict_test):
        parser.print_help()


if __name__ == "__main__":
    main()
