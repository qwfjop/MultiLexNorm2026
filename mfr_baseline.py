from __future__ import annotations

import argparse
import os

import pandas as pd

from utils import counting, evaluate, mfr, zip_files_flat


def _load_dataset(dataset_name: str, use_auth_token: bool = False):
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or (True if use_auth_token else None)
    return load_dataset(dataset_name, token=token)


def predict(train_dataset, test_dataset) -> pd.DataFrame:
    train_df = train_dataset.to_pandas()
    test_df = test_dataset.to_pandas()

    count_langs = {}
    for lang in train_df["lang"].unique():
        train_lang = train_df.loc[train_df["lang"] == lang]
        count_langs[lang] = counting(train_lang.to_dict(orient="records"))

    test_df["pred"] = test_df.apply(
        lambda row: mfr(row["raw"], count_langs.get(row["lang"], {})),
        axis=1,
    )
    return test_df


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
