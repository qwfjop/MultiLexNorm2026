from __future__ import annotations

import os
from pathlib import Path


def auth_token(use_auth_token: bool = False):
    return os.environ.get("HF_TOKEN") or (True if use_auth_token else None)


def load_dataset_auto(dataset_name_or_path: str, use_auth_token: bool = False):
    from datasets import load_dataset, load_from_disk

    path = Path(dataset_name_or_path)
    if path.exists():
        return load_from_disk(str(path))
    return load_dataset(dataset_name_or_path, token=auth_token(use_auth_token))


def cache_dataset(dataset_name: str, output_dir: str, use_auth_token: bool = False) -> str:
    data = load_dataset_auto(dataset_name, use_auth_token=use_auth_token)
    path = Path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.save_to_disk(str(path))
    return str(path)
