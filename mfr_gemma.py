from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import pandas as pd

from dataset_io import load_dataset_auto
from gemma_mlx_model import clean_prediction, format_prompt
from utils import counting, evaluate, gold_or_raw, mfr, zip_files_flat


DEFAULT_MODEL = "models/gemma4_mlx_filtered"
DEFAULT_ADAPTER = "models/gemma_mlx_balanced_lora_v1"


def _load_dataset(dataset_name: str, use_auth_token: bool = False):
    return load_dataset_auto(dataset_name, use_auth_token=use_auth_token)


def is_gemma_candidate(raw_token: str, mfr_prediction: str, counts: dict[str, dict[str, int]]) -> bool:
    if mfr_prediction != raw_token:
        return False
    if raw_token in counts:
        return False
    if len(raw_token) <= 1 or len(raw_token) > 40:
        return False
    if raw_token.startswith(("%", "#", "@", "http://", "https://", "www.")):
        return False
    if raw_token.isdigit():
        return False
    return any(char.isalnum() for char in raw_token)


def normalized_vocabulary(records: list[dict]) -> set[str]:
    vocab: set[str] = set()
    for record in records:
        for raw_token, norm_token in zip(record["raw"], record["norm"]):
            token = gold_or_raw(raw_token, norm_token)
            if token:
                vocab.add(token)
    return vocab


def acceptable_gemma_prediction(raw_token: str, prediction: str, norm_vocab: set[str]) -> bool:
    if prediction == raw_token:
        return False
    if prediction not in norm_vocab:
        return False
    if not prediction or len(prediction) > 60:
        return False
    if "\n" in prediction or "\t" in prediction:
        return False
    if prediction.startswith(("%", "#", "@")) and not raw_token.startswith(("%", "#", "@")):
        return False
    if prediction.lower().startswith(("thought", "normalized", "language:", "sentence")):
        return False
    return any(char.isalnum() for char in prediction)


def gemma_predictions_for_candidates(
    rows: pd.DataFrame,
    mfr_predictions: list[list[str]],
    count_langs: dict[str, dict[str, dict[str, int]]],
    norm_vocab_langs: dict[str, set[str]],
    model_path: str,
    adapter_path: str | None,
    batch_size: int,
    max_tokens: int,
) -> dict[tuple[int, int], str]:
    import mlx.core as mx
    from mlx_lm import batch_generate, load

    model, tokenizer = load(model_path, adapter_path=adapter_path)
    prompts: list[Sequence[int]] = []
    keys: list[tuple[int, int]] = []
    raw_tokens: list[str] = []

    for row_index, row in enumerate(rows.itertuples(index=False)):
        raw_sentence = list(row.raw)
        counts = count_langs.get(row.lang, {})
        for token_index, raw_token in enumerate(raw_sentence):
            if not is_gemma_candidate(raw_token, mfr_predictions[row_index][token_index], counts):
                continue
            prompt = format_prompt(raw_sentence, row.lang, token_index)
            messages = [{"role": "user", "content": prompt}]
            prompts.append(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
            keys.append((row_index, token_index))
            raw_tokens.append(raw_token)

    print(f"gemma candidates={len(prompts)}", flush=True)
    predictions: dict[tuple[int, int], str] = {}
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        response = batch_generate(
            model,
            tokenizer,
            prompts[start:end],
            verbose=False,
            max_tokens=max_tokens,
        )
        for key, text, raw_token in zip(keys[start:end], response.texts, raw_tokens[start:end]):
            prediction = clean_prediction(text, raw_token)
            if acceptable_gemma_prediction(raw_token, prediction, norm_vocab_langs.get(rows.iloc[key[0]]["lang"], set())):
                predictions[key] = prediction
        mx.clear_cache()
        if end == len(prompts) or end % 512 < batch_size:
            print(f"gemma predicted={end}/{len(prompts)} accepted={len(predictions)}", flush=True)

    return predictions


def predict(
    train_dataset,
    test_dataset,
    model_path: str,
    adapter_path: str | None,
    batch_size: int,
    max_tokens: int,
) -> pd.DataFrame:
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()

    count_langs: dict[str, dict[str, dict[str, int]]] = {}
    norm_vocab_langs: dict[str, set[str]] = {}
    for lang in train_df["lang"].unique():
        train_lang = train_df.loc[train_df["lang"] == lang]
        records = train_lang.to_dict(orient="records")
        count_langs[lang] = counting(records)
        norm_vocab_langs[lang] = normalized_vocabulary(records)

    mfr_predictions = [
        mfr(row.raw, count_langs.get(row.lang, {}))
        for row in test_df.itertuples(index=False)
    ]
    gemma_overrides = gemma_predictions_for_candidates(
        test_df,
        mfr_predictions,
        count_langs,
        norm_vocab_langs,
        model_path=model_path,
        adapter_path=adapter_path,
        batch_size=batch_size,
        max_tokens=max_tokens,
    )

    hybrid_predictions = [list(prediction) for prediction in mfr_predictions]
    for (row_index, token_index), prediction in gemma_overrides.items():
        hybrid_predictions[row_index][token_index] = prediction

    test_df["pred"] = hybrid_predictions
    return test_df


def evaluate_validation(
    dataset_name: str,
    model_path: str,
    adapter_path: str | None,
    use_auth_token: bool = False,
    batch_size: int = 16,
    max_tokens: int = 8,
    max_eval_examples: int | None = None,
    metrics_path: str | None = None,
) -> tuple[float, float, float]:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    validation = data["validation"]
    if max_eval_examples is not None:
        validation = validation.select(range(min(max_eval_examples, len(validation))))

    out = predict(
        data["train"],
        validation,
        model_path=model_path,
        adapter_path=adapter_path,
        batch_size=batch_size,
        max_tokens=max_tokens,
    )
    metrics = evaluate(
        raw=out["raw"].tolist(),
        gold=out["norm"].tolist(),
        pred=out["pred"].tolist(),
        info=True,
    )
    if metrics_path:
        payload = {
            "dataset": dataset_name,
            "model_path": model_path,
            "adapter_path": adapter_path,
            "max_eval_examples": max_eval_examples,
            "baseline_lai": metrics[0],
            "accuracy": metrics[1],
            "err": metrics[2],
        }
        path = Path(metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote metrics_path={path}", flush=True)
    return metrics


def create_submission(
    dataset_name: str,
    output_dir: str,
    model_path: str,
    adapter_path: str | None,
    use_auth_token: bool = False,
    batch_size: int = 16,
    max_tokens: int = 8,
    zip_output: bool = True,
) -> str:
    from datasets import concatenate_datasets

    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    train = concatenate_datasets([data["train"], data["validation"]])
    out = predict(
        train,
        data["test"],
        model_path=model_path,
        adapter_path=adapter_path,
        batch_size=batch_size,
        max_tokens=max_tokens,
    )

    os.makedirs(output_dir, exist_ok=True)
    prediction_path = os.path.join(output_dir, "predictions.json")
    out.to_json(prediction_path, orient="records")

    if zip_output:
        zip_files_flat(output_dir, f"{output_dir}.zip")

    return prediction_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a conservative MFR + Gemma hybrid.")
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", default="outputs/submission_mfr_gemma")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--use-auth-token", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--metrics-path")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_validation(
            args.dataset,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            max_eval_examples=args.max_eval_examples,
            metrics_path=args.metrics_path,
        )
    if args.predict_test:
        create_submission(
            args.dataset,
            args.output_dir,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            zip_output=not args.no_zip,
        )
    if not (args.eval_only or args.predict_test):
        parser.print_help()


if __name__ == "__main__":
    main()
