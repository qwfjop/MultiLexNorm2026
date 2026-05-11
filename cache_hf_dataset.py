from __future__ import annotations

import argparse

from dataset_io import cache_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset once and save it locally.")
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--output-dir", default="data/multilexnorm2026-dev-pub")
    parser.add_argument("--use-auth-token", action="store_true")
    args = parser.parse_args()

    output_dir = cache_dataset(
        args.dataset,
        args.output_dir,
        use_auth_token=args.use_auth_token,
    )
    print(f"saved dataset to {output_dir}", flush=True)
    print(f"use it later with: --dataset {output_dir}", flush=True)


if __name__ == "__main__":
    main()
