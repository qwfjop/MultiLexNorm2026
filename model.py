from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from typing import DefaultDict, Iterable, Mapping, Sequence

from dataset_io import load_dataset_auto
from utils import evaluate, gold_or_raw, zip_files_flat


TokenCounts = DefaultDict[str, Counter[str]]


def _new_token_counts() -> TokenCounts:
    return defaultdict(Counter)


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


def _best_replacement(
    counts: Mapping[str, Counter[str]],
    token: str,
    min_margin: int = 0,
) -> str | None:
    if token not in counts:
        return None
    choices = counts[token].most_common(2)
    replacement, top_count = choices[0]
    next_count = choices[1][1] if len(choices) > 1 else 0
    if top_count - next_count < min_margin:
        return None
    return replacement


class MFRPlusModel:
    """Per-language MFR with a conservative cross-language fallback."""

    def __init__(self, global_min_margin: int = 1, use_global_fallback: bool = True) -> None:
        self.global_min_margin = global_min_margin
        self.use_global_fallback = use_global_fallback
        self.lang_counts: DefaultDict[str, TokenCounts] = defaultdict(_new_token_counts)
        self.global_counts: TokenCounts = _new_token_counts()

    def fit(self, records: Iterable[Mapping[str, object]]) -> "MFRPlusModel":
        for record in records:
            lang = str(record["lang"])
            raw_tokens = _as_token_list(record["raw"])
            norm_tokens = _as_token_list(record["norm"])
            if raw_tokens is None or norm_tokens is None:
                continue

            for raw_token, norm_token in zip(raw_tokens, norm_tokens):
                target = gold_or_raw(raw_token, norm_token)

                self.lang_counts[lang][raw_token][target] += 1
                self.global_counts[raw_token][target] += 1

        return self

    def predict_token(self, token: str, lang: str) -> str:
        replacement = _best_replacement(self.lang_counts.get(lang, {}), token)
        if replacement is not None:
            return replacement

        if self.use_global_fallback:
            replacement = _best_replacement(
                self.global_counts,
                token,
                min_margin=self.global_min_margin,
            )
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


def train_model(
    train_dataset,
    global_min_margin: int = 1,
    use_global_fallback: bool = True,
) -> MFRPlusModel:
    return MFRPlusModel(
        global_min_margin=global_min_margin,
        use_global_fallback=use_global_fallback,
    ).fit(_fit_records(train_dataset))


def predict(
    train_dataset,
    test_dataset,
    global_min_margin: int = 1,
    use_global_fallback: bool = True,
):
    model = train_model(
        train_dataset,
        global_min_margin=global_min_margin,
        use_global_fallback=use_global_fallback,
    )
    return model.predict_dataframe(test_dataset.to_pandas())


def _load_dataset(dataset_name: str, use_auth_token: bool = False):
    return load_dataset_auto(dataset_name, use_auth_token=use_auth_token)


def evaluate_validation(
    dataset_name: str,
    use_auth_token: bool = False,
    global_min_margin: int = 1,
    use_global_fallback: bool = True,
) -> tuple[float, float, float]:
    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    model = train_model(
        data["train"],
        global_min_margin=global_min_margin,
        use_global_fallback=use_global_fallback,
    )
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
    global_min_margin: int = 1,
    use_global_fallback: bool = True,
) -> str:
    from datasets import concatenate_datasets

    data = _load_dataset(dataset_name, use_auth_token=use_auth_token)
    train = concatenate_datasets([data["train"], data["validation"]])
    out = predict(
        train,
        data["test"],
        global_min_margin=global_min_margin,
        use_global_fallback=use_global_fallback,
    )

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
    parser.add_argument(
        "--global-min-margin",
        type=int,
        default=1,
        help="Minimum global-count margin required before using a cross-language fallback.",
    )
    parser.add_argument(
        "--disable-global-fallback",
        action="store_true",
        help="Disable the cross-language fallback and run plain per-language MFR.",
    )
    args = parser.parse_args()

    if args.eval_only:
        evaluate_validation(
            args.dataset,
            use_auth_token=args.use_auth_token,
            global_min_margin=args.global_min_margin,
            use_global_fallback=not args.disable_global_fallback,
        )
    else:
        create_submission(
            args.dataset,
            args.output_dir,
            zip_output=not args.no_zip,
            use_auth_token=args.use_auth_token,
            global_min_margin=args.global_min_margin,
            use_global_fallback=not args.disable_global_fallback,
        )


if __name__ == "__main__":
    main()
