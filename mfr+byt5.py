from __future__ import annotations

import argparse
import os
from typing import Sequence

import pandas as pd

from byt5_model import ByT5Normalizer, _load_dataset
from utils import counting, evaluate, gold_or_raw, zip_files_flat


def build_mfr_counts_by_language(train_dataset) -> dict[str, dict[str, dict[str, int]]]:
    train_df = train_dataset.to_pandas()
    counts_by_lang = {}
    for lang in train_df["lang"].unique():
        train_lang = train_df.loc[train_df["lang"] == lang]
        counts_by_lang[lang] = counting(train_lang.to_dict(orient="records"))
    return counts_by_lang


def build_target_vocab_by_language(train_dataset) -> dict[str, set[str]]:
    target_vocab_by_lang: dict[str, set[str]] = {}
    for row in train_dataset:
        lang = row["lang"]
        target_vocab = target_vocab_by_lang.setdefault(lang, set())
        for raw_token, norm_token in zip(row["raw"], row["norm"]):
            target_vocab.add(gold_or_raw(raw_token, norm_token))
    return target_vocab_by_lang


def most_frequent_replacement(token: str, counts: dict[str, dict[str, int]]) -> str:
    if token not in counts:
        return token
    return max(counts[token], key=counts[token].get)


def is_safe_byt5_prediction(
    raw_token: str,
    prediction: str,
    target_vocab: set[str],
    max_extra_chars: int,
    require_known_output: bool,
) -> bool:
    if not prediction:
        return False
    if len(prediction) > len(raw_token) + max_extra_chars:
        return False
    if prediction == raw_token:
        return True
    if require_known_output and prediction not in target_vocab:
        return False
    return True


def predict_hybrid(
    train_dataset,
    test_dataset,
    byt5_model_dir: str,
    batch_size: int = 64,
    max_input_length: int = 128,
    max_new_tokens: int = 64,
    max_extra_chars: int = 10,
    require_known_output: bool = True,
    fallback_mode: str = "unknown-only",
) -> pd.DataFrame:
    counts_by_lang = build_mfr_counts_by_language(train_dataset)
    target_vocab_by_lang = build_target_vocab_by_language(train_dataset)
    model = ByT5Normalizer.from_pretrained(byt5_model_dir)
    out = test_dataset.to_pandas().copy()

    predictions: list[list[str]] = []
    byt5_items: list[tuple[str, str]] = []
    byt5_slots: list[tuple[int, int, str, str]] = []
    mfr_changed = 0
    byt5_candidates = 0

    for row_index, row in enumerate(out.itertuples(index=False)):
        lang = row.lang
        lang_counts = counts_by_lang.get(lang, {})
        row_predictions = []

        for token_index, raw_token in enumerate(row.raw):
            mfr_prediction = most_frequent_replacement(raw_token, lang_counts)

            should_try_byt5 = False
            if fallback_mode == "unknown-only":
                should_try_byt5 = raw_token not in lang_counts
            elif fallback_mode == "mfr-unchanged":
                should_try_byt5 = mfr_prediction == raw_token
            else:
                raise ValueError(f"Unsupported fallback mode: {fallback_mode}")

            if should_try_byt5:
                row_predictions.append(raw_token)
                byt5_items.append((raw_token, lang))
                byt5_slots.append((row_index, token_index, raw_token, lang))
                byt5_candidates += 1
            else:
                row_predictions.append(mfr_prediction)
                if mfr_prediction != raw_token:
                    mfr_changed += 1

        predictions.append(row_predictions)

    print(
        "hybrid "
        f"sentences={len(out)} byt5_candidates={byt5_candidates} "
        f"mfr_changed={mfr_changed} fallback_mode={fallback_mode}",
        flush=True,
    )

    if byt5_items:
        byt5_predictions = model.predict_token_items(
            byt5_items,
            batch_size=batch_size,
            max_input_length=max_input_length,
            max_new_tokens=max_new_tokens,
        )
        accepted = 0
        rejected = 0
        for (row_index, token_index, raw_token, lang), byt5_prediction in zip(byt5_slots, byt5_predictions):
            target_vocab = target_vocab_by_lang.get(lang, set())
            if is_safe_byt5_prediction(
                raw_token,
                byt5_prediction,
                target_vocab=target_vocab,
                max_extra_chars=max_extra_chars,
                require_known_output=require_known_output,
            ):
                predictions[row_index][token_index] = byt5_prediction
                accepted += 1
            else:
                predictions[row_index][token_index] = raw_token
                rejected += 1
        print(f"byt5 accepted={accepted} rejected={rejected}", flush=True)

    out["pred"] = predictions
    return out


def evaluate_validation(
    dataset_name: str,
    byt5_model_dir: str,
    use_auth_token: bool = False,
    batch_size: int = 64,
    max_extra_chars: int = 10,
    require_known_output: bool = True,
    fallback_mode: str = "unknown-only",
) -> tuple[float, float, float]:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    out = predict_hybrid(
        data["train"],
        data["validation"],
        byt5_model_dir=byt5_model_dir,
        batch_size=batch_size,
        max_extra_chars=max_extra_chars,
        require_known_output=require_known_output,
        fallback_mode=fallback_mode,
    )
    return evaluate(
        raw=out["raw"].tolist(),
        gold=out["norm"].tolist(),
        pred=out["pred"].tolist(),
        info=True,
    )


def inspect_validation_predictions(
    dataset_name: str,
    byt5_model_dir: str,
    use_auth_token: bool = False,
    batch_size: int = 64,
    limit: int = 10,
    langs: set[str] | None = None,
    max_extra_chars: int = 10,
    require_known_output: bool = True,
    fallback_mode: str = "unknown-only",
) -> None:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    validation = data["validation"]
    if langs is not None:
        validation = validation.filter(lambda row: row["lang"] in langs)

    out = predict_hybrid(
        data["train"],
        validation.select(range(min(limit, len(validation)))),
        byt5_model_dir=byt5_model_dir,
        batch_size=batch_size,
        max_extra_chars=max_extra_chars,
        require_known_output=require_known_output,
        fallback_mode=fallback_mode,
    )

    for index, row in enumerate(out.itertuples(index=False), start=1):
        print(f"\n[{index}] lang={row.lang}")
        print("raw :", row.raw)
        print("gold:", row.norm)
        print("pred:", row.pred)
        changed = [
            f"{raw} -> pred:{prediction} / gold:{gold}"
            for raw, gold, prediction in zip(row.raw, row.norm, row.pred)
            if prediction != gold
        ]
        print("diff:", "; ".join(changed) if changed else "all correct")


def create_submission(
    dataset_name: str,
    byt5_model_dir: str,
    output_dir: str,
    use_auth_token: bool = False,
    batch_size: int = 64,
    max_extra_chars: int = 10,
    require_known_output: bool = True,
    fallback_mode: str = "unknown-only",
    zip_output: bool = True,
) -> str:
    from datasets import concatenate_datasets

    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    train = concatenate_datasets([data["train"], data["validation"]])
    out = predict_hybrid(
        train,
        data["test"],
        byt5_model_dir=byt5_model_dir,
        batch_size=batch_size,
        max_extra_chars=max_extra_chars,
        require_known_output=require_known_output,
        fallback_mode=fallback_mode,
    )

    os.makedirs(output_dir, exist_ok=True)
    prediction_path = os.path.join(output_dir, "predictions.json")
    out.to_json(prediction_path, orient="records")

    if zip_output:
        zip_files_flat(output_dir, f"{output_dir}.zip")

    return prediction_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an MFR + ByT5 hybrid normalizer.")
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--model-dir", default="models/byt5_copyaware", help="Fine-tuned ByT5 checkpoint.")
    parser.add_argument("--output-dir", default="outputs/submission_mfr_byt5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-extra-chars", type=int, default=10)
    parser.add_argument(
        "--allow-unseen-byt5-output",
        action="store_true",
        help="Accept changed ByT5 outputs even when the generated token was not seen as a training target.",
    )
    parser.add_argument(
        "--fallback-mode",
        choices=["unknown-only", "mfr-unchanged"],
        default="unknown-only",
        help="unknown-only is safer; mfr-unchanged lets ByT5 try every token MFR would copy.",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--inspect-limit", type=int, default=10)
    parser.add_argument("--inspect-langs", help="Comma-separated language filter, e.g. en,ko")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--use-auth-token", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args(argv)

    if args.eval_only:
        evaluate_validation(
            args.dataset,
            args.model_dir,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
            max_extra_chars=args.max_extra_chars,
            require_known_output=not args.allow_unseen_byt5_output,
            fallback_mode=args.fallback_mode,
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
            max_extra_chars=args.max_extra_chars,
            require_known_output=not args.allow_unseen_byt5_output,
            fallback_mode=args.fallback_mode,
        )
    if args.predict_test:
        create_submission(
            args.dataset,
            args.model_dir,
            args.output_dir,
            use_auth_token=args.use_auth_token,
            batch_size=args.batch_size,
            max_extra_chars=args.max_extra_chars,
            require_known_output=not args.allow_unseen_byt5_output,
            fallback_mode=args.fallback_mode,
            zip_output=not args.no_zip,
        )
    if not (args.eval_only or args.inspect or args.predict_test):
        parser.print_help()


if __name__ == "__main__":
    main()
