from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from typing import DefaultDict, Iterable, Mapping, Sequence

from utils import evaluate, zip_files_flat


TokenCounts = DefaultDict[str, Counter[str]]


def _new_token_counts() -> TokenCounts:
    return defaultdict(Counter)


def _best_replacement(counts: Mapping[str, Counter[str]], token: str) -> str | None:
    if token not in counts:
        return None
    return counts[token].most_common(1)[0][0]


class MFRPlusModel:
    """Most-frequent replacement baseline with conservative fallback tables."""

    def __init__(self) -> None:
        self.lang_counts: DefaultDict[str, TokenCounts] = defaultdict(_new_token_counts)
        self.lang_lower_counts: DefaultDict[str, TokenCounts] = defaultdict(_new_token_counts)
        self.global_counts: TokenCounts = _new_token_counts()
        self.global_lower_counts: TokenCounts = _new_token_counts()

    def fit(self, records: Iterable[Mapping[str, object]]) -> "MFRPlusModel":
        for record in records:
            lang = str(record["lang"])
            raw_tokens = record["raw"]
            norm_tokens = record["norm"]
            if not isinstance(raw_tokens, Sequence) or not isinstance(norm_tokens, Sequence):
                continue

            for raw_token, norm_token in zip(raw_tokens, norm_tokens):
                if not isinstance(raw_token, str) or not isinstance(norm_token, str):
                    continue
                if norm_token == "":
                    continue

                self.lang_counts[lang][raw_token][norm_token] += 1
                self.global_counts[raw_token][norm_token] += 1

                lower_token = raw_token.lower()
                self.lang_lower_counts[lang][lower_token][norm_token] += 1
                self.global_lower_counts[lower_token][norm_token] += 1

        return self

    def predict_token(self, token: str, lang: str) -> str:
        for counts, key in (
            (self.lang_counts.get(lang, {}), token),
            (self.global_counts, token),
            (self.lang_lower_counts.get(lang, {}), token.lower()),
            (self.global_lower_counts, token.lower()),
        ):
            replacement = _best_replacement(counts, key)
            if replacement is not None:
                return replacement
        return token

    def predict_sentence(self, raw_tokens: Sequence[str], lang: str) -> list[str]:
        return [self.predict_token(token, lang) for token in raw_tokens]

    def predict_dataframe(self, data):
        out = data.copy()
        out["pred"] = out.apply(
            lambda row: self.predict_sentence(row["raw"], row["lang"]),
            axis=1,
        )
        return out


def _fit_records(dataset) -> list[dict]:
    return dataset.to_pandas().to_dict(orient="records")


def train_model(train_dataset) -> MFRPlusModel:
    return MFRPlusModel().fit(_fit_records(train_dataset))


def predict(train_dataset, test_dataset):
    model = train_model(train_dataset)
    return model.predict_dataframe(test_dataset.to_pandas())


def _load_dataset(dataset_name: str, use_auth_token: bool = False):
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or (True if use_auth_token else None)
    return load_dataset(dataset_name, token=token)


def evaluate_validation(dataset_name: str, use_auth_token: bool = False) -> tuple[float, float, float]:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    model = train_model(data["train"])
    out = model.predict_dataframe(data["validation"].to_pandas())
    return evaluate(
        raw=out["raw"].tolist(),
        gold=out["norm"].tolist(),
        pred=out["pred"].tolist(),
        info=True,
    )


def create_submission(
    dataset_name: str,
    output_dir: str,
    zip_output: bool = True,
    use_auth_token: bool = False,
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
    parser = argparse.ArgumentParser(description="Train and run a lightweight MultiLexNorm model.")
    parser.add_argument(
        "--dataset",
        default="weerayut/multilexnorm2026-dev-pub",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/submission_dev",
        help="Directory where predictions.json will be written.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate on the validation split instead of writing test predictions.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not create a zip file next to the output directory.",
    )
    parser.add_argument(
        "--use-auth-token",
        action="store_true",
        help="Use the logged-in Hugging Face token, or the HF_TOKEN environment variable.",
    )
    args = parser.parse_args()

    if args.eval_only:
        evaluate_validation(args.dataset, use_auth_token=args.use_auth_token)
    else:
        create_submission(
            args.dataset,
            args.output_dir,
            zip_output=not args.no_zip,
            use_auth_token=args.use_auth_token,
        )


if __name__ == "__main__":
    main()
