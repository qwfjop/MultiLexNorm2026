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


def iter_token_examples(records: Iterable[Mapping[str, object]]):
    for record in records:
        lang = str(record["lang"])
        raw_tokens = _as_token_list(record["raw"])
        norm_tokens = _as_token_list(record["norm"])
        if raw_tokens is None or norm_tokens is None:
            continue

        for raw_token, norm_token in zip(raw_tokens, norm_tokens):
            yield {
                "input": format_input(raw_token, lang),
                "target": gold_or_raw(raw_token, norm_token),
            }


def format_input(token: str, lang: str) -> str:
    return f"normalize lang={lang}: {token}"


class TokenNormalizationDataset:
    def __init__(self, examples, tokenizer, max_input_length: int, max_target_length: int) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        model_input = self.tokenizer(
            example["input"],
            max_length=self.max_input_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=example["target"],
            max_length=self.max_target_length,
            truncation=True,
        )
        model_input["labels"] = labels["input_ids"]
        return model_input


@dataclass
class ByT5TrainConfig:
    model_name: str = "google/byt5-small"
    output_dir: str = "models/byt5"
    max_input_length: int = 128
    max_target_length: int = 64
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    epochs: int = 1
    max_train_examples: int | None = None
    log_every_steps: int = 50


class ByT5Normalizer:
    def __init__(self, model_name: str = "google/byt5-small", device=None, tokenizer=None, model=None) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = model or AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device or _select_device()
        self.model.to(self.device)

    def fit(self, records: Iterable[Mapping[str, object]], config: ByT5TrainConfig) -> "ByT5Normalizer":
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader
        from transformers import DataCollatorForSeq2Seq

        examples = list(iter_token_examples(records))
        if config.max_train_examples is not None:
            examples = examples[: config.max_train_examples]
        if not examples:
            raise ValueError("No training examples were produced from the dataset.")

        dataset = TokenNormalizationDataset(
            examples,
            tokenizer=self.tokenizer,
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
        )
        collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )

        print(
            "training "
            f"examples={len(dataset)} batches_per_epoch={len(loader)} "
            f"batch_size={config.batch_size} grad_accum={config.gradient_accumulation_steps} "
            f"device={self.device}",
            flush=True,
        )

        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
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

        os.makedirs(config.output_dir, exist_ok=True)
        self.model.save_pretrained(config.output_dir)
        self.tokenizer.save_pretrained(config.output_dir)
        return self

    @classmethod
    def from_pretrained(cls, model_path: str, device=None) -> "ByT5Normalizer":
        return cls(model_path, device=device)

    @classmethod
    def tiny_random(cls, device=None) -> "ByT5Normalizer":
        from transformers import ByT5Tokenizer, T5Config, T5ForConditionalGeneration

        tokenizer = ByT5Tokenizer()
        config = T5Config(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = T5ForConditionalGeneration(config)
        return cls("local-tiny-random-byt5", device=device, tokenizer=tokenizer, model=model)

    def predict_tokens(
        self,
        raw_tokens: Sequence[str],
        lang: str,
        batch_size: int = 16,
        max_input_length: int = 128,
        max_new_tokens: int = 64,
    ) -> list[str]:
        predictions: list[str] = []
        self.model.eval()

        import torch

        with torch.no_grad():
            for start in range(0, len(raw_tokens), batch_size):
                chunk = list(raw_tokens[start : start + batch_size])
                inputs = [format_input(token, lang) for token in chunk]
                encoded = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                )
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                predictions.extend(text.strip() if text.strip() else token for text, token in zip(decoded, chunk))

        return predictions

    def predict_token_items(
        self,
        items: Sequence[tuple[str, str]],
        batch_size: int = 16,
        max_input_length: int = 128,
        max_new_tokens: int = 64,
        log_every_batches: int = 25,
    ) -> list[str]:
        predictions: list[str] = []
        self.model.eval()

        import torch

        total_batches = (len(items) + batch_size - 1) // batch_size
        print(
            f"predicting tokens={len(items)} batch_size={batch_size} batches={total_batches}",
            flush=True,
        )

        with torch.inference_mode():
            for batch_number, start in enumerate(range(0, len(items), batch_size), start=1):
                chunk = list(items[start : start + batch_size])
                inputs = [format_input(token, lang) for token, lang in chunk]
                encoded = self.tokenizer(
                    inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_input_length,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )
                decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
                predictions.extend(
                    text.strip() if text.strip() else token
                    for text, (token, _) in zip(decoded, chunk)
                )

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
        batch_size: int = 16,
        max_input_length: int = 128,
        max_new_tokens: int = 64,
    ) -> pd.DataFrame:
        out = data.copy()
        token_items: list[tuple[str, str]] = []
        row_lengths: list[int] = []

        for row in out.itertuples(index=False):
            raw_tokens = list(row.raw)
            row_lengths.append(len(raw_tokens))
            token_items.extend((token, row.lang) for token in raw_tokens)

        flat_predictions = self.predict_token_items(
            token_items,
            batch_size=batch_size,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )

        preds: list[list[str]] = []
        cursor = 0
        for row_length in row_lengths:
            preds.append(flat_predictions[cursor : cursor + row_length])
            cursor += row_length

        out["pred"] = preds
        return out


def _records_from_dataset(dataset) -> list[dict]:
    return list(dataset)


def train_from_hf_dataset(
    dataset_name: str,
    config: ByT5TrainConfig,
    use_auth_token: bool = False,
    cli_args: Sequence[str] | None = None,
) -> ByT5Normalizer:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    model = ByT5Normalizer(config.model_name)
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
    config: ByT5TrainConfig,
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
    batch_size: int = 16,
) -> tuple[float, float, float]:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    model = ByT5Normalizer.from_pretrained(model_path)
    out = model.predict_dataframe(data["validation"].to_pandas(), batch_size=batch_size)
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
    batch_size: int = 16,
    limit: int = 10,
    langs: set[str] | None = None,
) -> None:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    model = ByT5Normalizer.from_pretrained(model_path)

    shown = 0
    for row in data["validation"]:
        if langs is not None and row["lang"] not in langs:
            continue
        shown += 1
        pred = model.predict_tokens(row["raw"], row["lang"], batch_size=batch_size)
        print(f"\n[{shown}] lang={row['lang']}")
        print("raw :", row["raw"])
        print("gold:", row["norm"])
        print("pred:", pred)

        changed = [
            f"{raw} -> pred:{prediction} / gold:{gold}"
            for raw, gold, prediction in zip(row["raw"], row["norm"], pred)
            if prediction != gold
        ]
        if changed:
            print("diff:", "; ".join(changed))
        else:
            print("diff: all correct")

        if shown >= limit:
            break

    if shown == 0:
        print("No validation examples matched the requested language filter.")


def create_submission(
    dataset_name: str,
    model_path: str,
    output_dir: str,
    use_auth_token: bool = False,
    batch_size: int = 16,
    zip_output: bool = True,
) -> str:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    model = ByT5Normalizer.from_pretrained(model_path)
    out = model.predict_dataframe(data["test"].to_pandas(), batch_size=batch_size)

    os.makedirs(output_dir, exist_ok=True)
    prediction_path = os.path.join(output_dir, "predictions.json")
    out.to_json(prediction_path, orient="records")

    if zip_output:
        zip_files_flat(output_dir, f"{output_dir}.zip")

    return prediction_path


def run_smoke_test(model_name: str, output_dir: str) -> None:
    records = [
        {"lang": "en", "raw": ["bcause", "u", "r"], "norm": ["because", "you", "are"]},
        {"lang": "en", "raw": ["bcause", "u", "r"], "norm": ["because", "you", "are"]},
        {"lang": "vi", "raw": ["ko", "mn"], "norm": ["không", "mọi người"]},
    ]
    config = ByT5TrainConfig(
        model_name=model_name,
        output_dir=output_dir,
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        epochs=1,
        max_train_examples=6,
        log_every_steps=1,
    )
    if model_name == "local-tiny-random-byt5":
        model = ByT5Normalizer.tiny_random()
    else:
        model = ByT5Normalizer(model_name)
    model.fit(records, config)
    pred = model.predict_tokens(["bcause", "u"], "en", batch_size=2)
    print("smoke prediction:", pred)
    if len(pred) != 2:
        raise AssertionError("ByT5 smoke prediction did not preserve token count.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune ByT5 for MultiLexNorm token normalization.")
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--model-name", default="google/byt5-small")
    parser.add_argument("--model-dir", default="models/byt5")
    parser.add_argument("--output-dir", default="outputs/submission_byt5")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-input-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--log-every-steps", type=int, default=50)
    parser.add_argument("--use-auth-token", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--inspect-limit", type=int, default=10)
    parser.add_argument("--inspect-langs", help="Comma-separated language filter, e.g. en,ko")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(args.model_name, args.model_dir)
        return

    config = ByT5TrainConfig(
        model_name=args.model_name,
        output_dir=args.model_dir,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_train_examples=args.max_train_examples,
        log_every_steps=args.log_every_steps,
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
        )
    if args.predict_test:
        create_submission(
            args.dataset,
            args.model_dir,
            args.output_dir,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
            zip_output=not args.no_zip,
        )
    if not (args.train or args.eval_only or args.inspect or args.predict_test):
        parser.print_help()


if __name__ == "__main__":
    main()
