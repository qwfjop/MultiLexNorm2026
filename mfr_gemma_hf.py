from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from dataset_io import load_dataset_auto
from gemma_model import GemmaNormalizer
from utils import counting, evaluate, gold_or_raw, mfr, zip_files_flat


DEFAULT_MODEL_DIR = "models/gemma_colab_lora"


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
    lowered = prediction.lower()
    if lowered.startswith(("thought", "normalized", "language:", "sentence", "target")):
        return False
    return any(char.isalnum() for char in prediction)


def predict(
    train_dataset,
    test_dataset,
    model_dir: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    candidate_limit: int | None = None,
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

    candidate_items: list[tuple[list[str], str, int]] = []
    candidate_keys: list[tuple[int, int]] = []
    candidate_raw: list[str] = []
    for row_index, row in enumerate(test_df.itertuples(index=False)):
        raw_sentence = list(row.raw)
        counts = count_langs.get(row.lang, {})
        for token_index, raw_token in enumerate(raw_sentence):
            if not is_gemma_candidate(raw_token, mfr_predictions[row_index][token_index], counts):
                continue
            candidate_items.append((raw_sentence, row.lang, token_index))
            candidate_keys.append((row_index, token_index))
            candidate_raw.append(raw_token)
            if candidate_limit is not None and len(candidate_items) >= candidate_limit:
                break
        if candidate_limit is not None and len(candidate_items) >= candidate_limit:
            break

    print(f"gemma candidates={len(candidate_items)}", flush=True)
    gemma_overrides: dict[tuple[int, int], str] = {}
    if candidate_items:
        model = GemmaNormalizer.from_pretrained(model_dir)
        raw_predictions = model.predict_token_items(
            candidate_items,
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            log_every_batches=10,
        )
        for key, raw_token, prediction in zip(candidate_keys, candidate_raw, raw_predictions):
            lang = test_df.iloc[key[0]]["lang"]
            if acceptable_gemma_prediction(raw_token, prediction, norm_vocab_langs.get(lang, set())):
                gemma_overrides[key] = prediction
        print(
            f"gemma predicted={len(raw_predictions)}/{len(candidate_items)} accepted={len(gemma_overrides)}",
            flush=True,
        )

    hybrid_predictions = [list(prediction) for prediction in mfr_predictions]
    for (row_index, token_index), prediction in gemma_overrides.items():
        hybrid_predictions[row_index][token_index] = prediction

    test_df["pred"] = hybrid_predictions
    return test_df


def evaluate_validation(
    dataset_name: str,
    model_dir: str,
    use_auth_token: bool = False,
    batch_size: int = 8,
    max_length: int = 256,
    max_new_tokens: int = 8,
    max_eval_examples: int | None = None,
    candidate_limit: int | None = None,
    metrics_path: str | None = None,
    predictions_path: str | None = None,
) -> tuple[float, float, float]:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    validation = data["validation"]
    if max_eval_examples is not None:
        validation = validation.select(range(min(max_eval_examples, len(validation))))

    out = predict(
        data["train"],
        validation,
        model_dir=model_dir,
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        candidate_limit=candidate_limit,
    )
    metrics = evaluate(
        raw=out["raw"].tolist(),
        gold=out["norm"].tolist(),
        pred=out["pred"].tolist(),
        info=True,
    )

    if predictions_path:
        pred_path = Path(predictions_path)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "lang": row.lang,
                "raw": list(row.raw),
                "gold": list(row.norm),
                "pred": list(row.pred),
            }
            for row in out.itertuples(index=False)
        ]
        pred_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote predictions_path={pred_path}", flush=True)

    if metrics_path:
        payload = {
            "dataset": dataset_name,
            "model_dir": model_dir,
            "max_eval_examples": max_eval_examples,
            "candidate_limit": candidate_limit,
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
    model_dir: str,
    use_auth_token: bool = False,
    batch_size: int = 8,
    max_length: int = 256,
    max_new_tokens: int = 8,
    zip_output: bool = True,
) -> str:
    from datasets import concatenate_datasets

    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    train = concatenate_datasets([data["train"], data["validation"]])
    out = predict(
        train,
        data["test"],
        model_dir=model_dir,
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
    parser = argparse.ArgumentParser(description="Run a conservative MFR + Hugging Face Gemma hybrid.")
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default="outputs/submission_mfr_gemma_hf")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--use-auth-token", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-eval-examples", type=int, default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--metrics-path")
    parser.add_argument("--predictions-path")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_validation(
            args.dataset,
            model_dir=args.model_dir,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            max_eval_examples=args.max_eval_examples,
            candidate_limit=args.candidate_limit,
            metrics_path=args.metrics_path,
            predictions_path=args.predictions_path,
        )
    if args.predict_test:
        create_submission(
            args.dataset,
            args.output_dir,
            model_dir=args.model_dir,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            zip_output=not args.no_zip,
        )
    if not (args.eval_only or args.predict_test):
        parser.print_help()


if __name__ == "__main__":
    main()
