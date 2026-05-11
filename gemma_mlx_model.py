from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from dataset_io import load_dataset_auto
from utils import evaluate, gold_or_raw


DEFAULT_MODEL = "models/gemma4_mlx_filtered"
DEFAULT_ADAPTER = "models/gemma_mlx_adapters"


def _load_dataset(dataset_name: str, use_auth_token: bool = False):
    return load_dataset_auto(dataset_name, use_auth_token=use_auth_token)


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


def format_sentence_prompt(raw_tokens: Sequence[str], lang: str) -> str:
    return (
        "Normalize this tokenized sentence.\n"
        "Return only a JSON array of normalized tokens with the same length. Do not explain.\n\n"
        f"Language: {lang}\n"
        f"Sentence tokens: {json.dumps(list(raw_tokens), ensure_ascii=False)}\n"
        "Normalized tokens:"
    )


def iter_context_token_examples(records: Iterable[Mapping[str, object]]):
    for record in records:
        lang = str(record["lang"])
        raw_tokens = _as_token_list(record["raw"])
        norm_tokens = _as_token_list(record["norm"])
        if raw_tokens is None or norm_tokens is None:
            continue

        for target_index, (raw_token, norm_token) in enumerate(zip(raw_tokens, norm_tokens)):
            target = gold_or_raw(raw_token, norm_token)
            if target == "":
                continue
            yield {
                "prompt": format_prompt(raw_tokens, lang, target_index),
                "completion": target,
                "changed": raw_token != target,
            }


def iter_sentence_examples(records: Iterable[Mapping[str, object]]):
    for record in records:
        lang = str(record["lang"])
        raw_tokens = _as_token_list(record["raw"])
        norm_tokens = _as_token_list(record["norm"])
        if raw_tokens is None or norm_tokens is None:
            continue
        targets = [gold_or_raw(raw_token, norm_token) for raw_token, norm_token in zip(raw_tokens, norm_tokens)]
        if len(targets) != len(raw_tokens) or any(target == "" for target in targets):
            continue
        yield {
            "prompt": format_sentence_prompt(raw_tokens, lang),
            "completion": json.dumps(targets, ensure_ascii=False),
            "changed": raw_tokens != targets,
        }


def balance_examples(
    examples: Sequence[Mapping[str, object]],
    copy_ratio: float | None,
    seed: int,
) -> list[Mapping[str, str]]:
    changed = [example for example in examples if example["changed"]]
    copied = [example for example in examples if not example["changed"]]
    rng = random.Random(seed)
    rng.shuffle(changed)
    rng.shuffle(copied)

    if copy_ratio is None:
        selected = list(examples)
    else:
        copy_limit = int(len(changed) * copy_ratio)
        selected = changed + copied[:copy_limit]

    rng.shuffle(selected)
    return [
        {"prompt": str(example["prompt"]), "completion": str(example["completion"])}
        for example in selected
    ]


def write_jsonl(path: Path, examples: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def prepare_mlx_data(
    dataset_name: str,
    data_dir: str,
    use_auth_token: bool = False,
    max_train_examples: int | None = None,
    max_valid_examples: int | None = 512,
    copy_ratio: float | None = None,
    seed: int = 13,
    example_mode: str = "token",
) -> None:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    iterator = iter_sentence_examples if example_mode == "sentence" else iter_context_token_examples
    raw_train_examples = list(iterator(data["train"]))
    raw_valid_examples = list(iterator(data["validation"]))
    train_changed = sum(1 for example in raw_train_examples if example["changed"])
    valid_changed = sum(1 for example in raw_valid_examples if example["changed"])
    train_examples = balance_examples(raw_train_examples, copy_ratio=copy_ratio, seed=seed)
    valid_examples = balance_examples(raw_valid_examples, copy_ratio=copy_ratio, seed=seed)

    if max_train_examples is not None:
        train_examples = train_examples[:max_train_examples]
    if max_valid_examples is not None:
        valid_examples = valid_examples[:max_valid_examples]

    out_dir = Path(data_dir)
    write_jsonl(out_dir / "train.jsonl", train_examples)
    write_jsonl(out_dir / "valid.jsonl", valid_examples)
    write_jsonl(out_dir / "test.jsonl", valid_examples)
    print(
        (
            f"wrote data_dir={out_dir} train={len(train_examples)} valid={len(valid_examples)} "
            f"train_changed={train_changed}/{len(raw_train_examples)} "
            f"valid_changed={valid_changed}/{len(raw_valid_examples)} "
            f"copy_ratio={copy_ratio} example_mode={example_mode}"
        ),
        flush=True,
    )


def train_mlx_lora(
    model_path: str,
    data_dir: str,
    adapter_path: str,
    iters: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
    num_layers: int,
    grad_accumulation_steps: int,
    steps_per_report: int,
    steps_per_eval: int,
    save_every: int,
    mask_prompt: bool,
    resume_adapter_file: str | None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model_path,
        "--train",
        "--data",
        data_dir,
        "--adapter-path",
        adapter_path,
        "--fine-tune-type",
        "lora",
        "--iters",
        str(iters),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--max-seq-length",
        str(max_seq_length),
        "--num-layers",
        str(num_layers),
        "--grad-accumulation-steps",
        str(grad_accumulation_steps),
        "--steps-per-report",
        str(steps_per_report),
        "--steps-per-eval",
        str(steps_per_eval),
        "--save-every",
        str(save_every),
        "--val-batches",
        "5",
    ]
    if mask_prompt:
        cmd.append("--mask-prompt")
    if resume_adapter_file:
        cmd.extend(["--resume-adapter-file", resume_adapter_file])
    print("running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def clean_prediction(text: str, raw_token: str) -> str:
    text = text.strip()
    for marker in ("<|channel|>", "<|channel>", "<|message|>", "<start_of_turn>", "<end_of_turn>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    if "\n" in text:
        text = text.splitlines()[0].strip()
    if text.startswith("Normalized token:"):
        text = text[len("Normalized token:") :].strip()
    if text.startswith("thought"):
        text = ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return text.strip() or raw_token


def clean_generation_text(text: str) -> str:
    text = text.strip()
    for marker in ("<|channel|>", "<|channel>", "<|message|>", "<start_of_turn>", "<end_of_turn>"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text.strip()


def parse_json_token_list(text: str, raw_tokens: Sequence[str]) -> list[str]:
    text = clean_generation_text(text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return list(raw_tokens)
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return list(raw_tokens)
    if not isinstance(parsed, list) or len(parsed) != len(raw_tokens):
        return list(raw_tokens)
    if not all(isinstance(token, str) and token != "" for token in parsed):
        return list(raw_tokens)
    return parsed


def predict_dataframe(
    data: pd.DataFrame,
    model_path: str,
    adapter_path: str | None,
    max_tokens: int,
    batch_size: int,
    use_chat_template: bool,
) -> pd.DataFrame:
    import mlx.core as mx
    from mlx_lm import batch_generate, load

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    out = data.copy()
    flat_prompts: list[str] = []
    flat_raw_tokens: list[str] = []
    row_lengths: list[int] = []
    total = sum(len(row.raw) for row in out.itertuples(index=False))
    done = 0

    for row in out.itertuples(index=False):
        raw_tokens = list(row.raw)
        row_lengths.append(len(raw_tokens))
        for target_index, raw_token in enumerate(raw_tokens):
            prompt = format_prompt(raw_tokens, row.lang, target_index)
            if use_chat_template:
                messages = [{"role": "user", "content": prompt}]
                prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            else:
                prompt = tokenizer.encode(prompt)
            flat_prompts.append(prompt)
            flat_raw_tokens.append(raw_token)

    flat_predictions: list[str] = []
    for start in range(0, len(flat_prompts), batch_size):
        end = min(start + batch_size, len(flat_prompts))
        response = batch_generate(
            model,
            tokenizer,
            flat_prompts[start:end],
            verbose=False,
            max_tokens=max_tokens,
        )
        for text, raw_token in zip(response.texts, flat_raw_tokens[start:end]):
            flat_predictions.append(clean_prediction(text, raw_token))
        done = end
        mx.clear_cache()
        if done == total or done % 512 < batch_size:
            print(f"predicted tokens={done}/{total}", flush=True)

    preds: list[list[str]] = []
    cursor = 0
    for row_length in row_lengths:
        preds.append(flat_predictions[cursor : cursor + row_length])
        cursor += row_length

    out["pred"] = preds
    return out


def predict_sentence_dataframe(
    data: pd.DataFrame,
    model_path: str,
    adapter_path: str | None,
    max_tokens: int,
    batch_size: int,
    use_chat_template: bool,
) -> pd.DataFrame:
    import mlx.core as mx
    from mlx_lm import batch_generate, load

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    out = data.copy()
    prompts: list[Sequence[int]] = []
    raw_rows: list[list[str]] = []
    total = len(out)

    for row in out.itertuples(index=False):
        raw_tokens = list(row.raw)
        prompt = format_sentence_prompt(raw_tokens, row.lang)
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        else:
            prompt = tokenizer.encode(prompt)
        prompts.append(prompt)
        raw_rows.append(raw_tokens)

    preds: list[list[str]] = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        response = batch_generate(
            model,
            tokenizer,
            prompts[start:end],
            verbose=False,
            max_tokens=max_tokens,
        )
        for text, raw_tokens in zip(response.texts, raw_rows[start:end]):
            preds.append(parse_json_token_list(text, raw_tokens))
        mx.clear_cache()
        if end == total or end % 128 < batch_size:
            print(f"predicted sentences={end}/{total}", flush=True)

    out["pred"] = preds
    return out


def evaluate_validation(
    dataset_name: str,
    model_path: str,
    adapter_path: str | None,
    use_auth_token: bool = False,
    max_eval_examples: int | None = 20,
    max_tokens: int = 8,
    batch_size: int = 8,
    use_chat_template: bool = True,
    example_mode: str = "token",
    predictions_path: str | None = None,
    eval_start: int = 0,
) -> tuple[float, float, float]:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    validation = data["validation"]
    if eval_start < 0:
        raise ValueError("--eval-start must be non-negative")
    if eval_start:
        validation = validation.select(range(eval_start, len(validation)))
    if max_eval_examples is not None:
        validation = validation.select(range(min(max_eval_examples, len(validation))))

    validation_df = validation.to_pandas()
    if example_mode == "sentence":
        out = predict_sentence_dataframe(
            validation_df,
            model_path=model_path,
            adapter_path=adapter_path,
            max_tokens=max_tokens,
            batch_size=batch_size,
            use_chat_template=use_chat_template,
        )
    else:
        out = predict_dataframe(
            validation_df,
            model_path=model_path,
            adapter_path=adapter_path,
            max_tokens=max_tokens,
            batch_size=batch_size,
            use_chat_template=use_chat_template,
        )
    if predictions_path:
        path = Path(predictions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for row in out.itertuples(index=False):
            rows.append(
                {
                    "lang": row.lang,
                    "raw": list(row.raw),
                    "gold": list(row.norm),
                    "pred": list(row.pred),
                }
            )
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote predictions_path={path}", flush=True)
    return evaluate(
        raw=out["raw"].tolist(),
        gold=out["norm"].tolist(),
        pred=out["pred"].tolist(),
        info=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate Gemma with MLX-LM LoRA.")
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default="models/gemma_mlx_data")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--use-auth-token", action="store_true")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-train-examples", type=int, default=1000)
    parser.add_argument("--max-valid-examples", type=int, default=512)
    parser.add_argument(
        "--copy-ratio",
        type=float,
        default=None,
        help="When set, train on all changed-token examples plus this many copy examples per changed example.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--example-mode", choices=["token", "sentence"], default="token")
    parser.add_argument("--max-eval-examples", type=int, default=20)
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--mask-prompt", action="store_true")
    parser.add_argument("--resume-adapter-file")
    parser.add_argument("--steps-per-report", type=int, default=5)
    parser.add_argument("--steps-per-eval", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--metrics-path")
    parser.add_argument("--predictions-path")
    args = parser.parse_args()
    train_adapter_path = args.adapter_path or DEFAULT_ADAPTER
    max_eval_examples = None if args.max_eval_examples <= 0 else args.max_eval_examples

    if args.prepare_data or args.train:
        prepare_mlx_data(
            args.dataset,
            args.data_dir,
            use_auth_token=args.use_auth_token,
            max_train_examples=args.max_train_examples,
            max_valid_examples=args.max_valid_examples,
            copy_ratio=args.copy_ratio,
            seed=args.seed,
            example_mode=args.example_mode,
        )
    if args.train:
        train_mlx_lora(
            model_path=args.model_path,
            data_dir=args.data_dir,
            adapter_path=train_adapter_path,
            iters=args.iters,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            num_layers=args.num_layers,
            grad_accumulation_steps=args.gradient_accumulation_steps,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            save_every=args.save_every,
            mask_prompt=args.mask_prompt,
            resume_adapter_file=args.resume_adapter_file,
        )
    if args.eval_only:
        metrics = evaluate_validation(
            args.dataset,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            use_auth_token=args.use_auth_token,
            max_eval_examples=max_eval_examples,
            max_tokens=args.max_tokens,
            batch_size=args.eval_batch_size,
            use_chat_template=not args.no_chat_template,
            example_mode=args.example_mode,
            predictions_path=args.predictions_path,
            eval_start=args.eval_start,
        )
        if args.metrics_path:
            payload = {
                "dataset": args.dataset,
                "model_path": args.model_path,
                "adapter_path": args.adapter_path,
                "max_eval_examples": max_eval_examples,
                "eval_start": args.eval_start,
                "baseline_lai": metrics[0],
                "accuracy": metrics[1],
                "err": metrics[2],
                "use_chat_template": not args.no_chat_template,
                "example_mode": args.example_mode,
            }
            metrics_path = Path(args.metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"wrote metrics_path={metrics_path}", flush=True)
    if not (args.prepare_data or args.train or args.eval_only):
        parser.print_help()


if __name__ == "__main__":
    main()
