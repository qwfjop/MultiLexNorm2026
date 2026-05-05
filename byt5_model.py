from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from utils import evaluate, zip_files_flat


def _load_dataset(dataset_name: str, use_auth_token: bool = False):
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or (True if use_auth_token else None)
    return load_dataset(dataset_name, token=token)


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
            if norm_token == "":
                continue
            yield {
                "input": format_input(raw_token, lang),
                "target": norm_token,
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

    def predict_dataframe(
        self,
        data: pd.DataFrame,
        batch_size: int = 16,
        max_input_length: int = 128,
        max_new_tokens: int = 64,
    ) -> pd.DataFrame:
        out = data.copy()
        out["pred"] = out.apply(
            lambda row: self.predict_tokens(
                row["raw"],
                row["lang"],
                batch_size=batch_size,
                max_input_length=max_input_length,
                max_new_tokens=max_new_tokens,
            ),
            axis=1,
        )
        return out


def _records_from_dataset(dataset) -> list[dict]:
    return list(dataset)


def train_from_hf_dataset(
    dataset_name: str,
    config: ByT5TrainConfig,
    use_auth_token: bool = False,
) -> ByT5Normalizer:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    model = ByT5Normalizer(config.model_name)
    return model.fit(_records_from_dataset(data["train"]), config)


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
        train_from_hf_dataset(args.dataset, config, use_auth_token=args.use_auth_token)
    if args.eval_only:
        evaluate_validation(
            args.dataset,
            args.model_dir,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
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
    if not (args.train or args.eval_only or args.predict_test):
        parser.print_help()


if __name__ == "__main__":
    main()
