from __future__ import annotations

import argparse
import os

import pandas as pd

from dataset_io import load_dataset_auto
from utils import evaluate, prediction_mfr_by_language, zip_files_flat


def _load_dataset(dataset_name: str, use_auth_token: bool = False):
    return load_dataset_auto(dataset_name, use_auth_token=use_auth_token)


def predict(train_dataset, test_dataset) -> pd.DataFrame:
    return prediction_mfr_by_language(train_dataset, test_dataset)


def evaluate_validation(dataset_name: str, use_auth_token: bool = False):
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    out = predict(data["train"], data["validation"])
    return evaluate(
        raw=out["raw"].tolist(),
        gold=out["norm"].tolist(),
        pred=out["pred"].tolist(),
        info=True,
    )


def create_submission(
    dataset_name: str,
    output_dir: str,
    use_auth_token: bool = False,
    zip_output: bool = True,
) -> str:
    from datasets import concatenate_datasets

    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    train = concatenate_datasets([data["train"], data["validation"]])
    out = predict(train, data["test"])

    os.makedirs(output_dir, exist_ok=True)
    prediction_path = os.path.join(output_dir, "predictions.json")
    out.to_json(prediction_path, orient="records")

    if zip_output:
        zip_files_flat(output_dir, f"{output_dir}.zip")

    return prediction_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the original MultiLexNorm MFR baseline.")
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--output-dir", default="outputs/submission_mfr")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--use-auth-token", action="store_true")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        evaluate_validation(args.dataset, use_auth_token=args.use_auth_token)
    if args.predict_test:
        create_submission(
            args.dataset,
            args.output_dir,
            use_auth_token=args.use_auth_token,
            zip_output=not args.no_zip,
        )
    if not (args.eval_only or args.predict_test):
        parser.print_help()


if __name__ == "__main__":
    main()
