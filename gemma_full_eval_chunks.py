from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset

from utils import evaluate


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable full validation for Gemma MLX.")
    parser.add_argument("--dataset", default="weerayut/multilexnorm2026-dev-pub")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--out-dir", default="outputs/gemma_full_eval_chunks")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--example-mode", choices=["token", "sentence"], default="token")
    parser.add_argument("--use-auth-token", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    args = parser.parse_args()

    token = True if args.use_auth_token else None
    validation = load_dataset(args.dataset, token=token)["validation"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(validation), args.chunk_size):
        count = min(args.chunk_size, len(validation) - start)
        pred_path = out_dir / f"pred_{start:05d}_{start + count:05d}.json"
        metrics_path = out_dir / f"metrics_{start:05d}_{start + count:05d}.json"
        if pred_path.exists() and metrics_path.exists():
            print(f"skip chunk start={start} count={count}", flush=True)
            continue

        cmd = [
            sys.executable,
            "gemma_mlx_model.py",
            "--eval-only",
            "--dataset",
            args.dataset,
            "--model-path",
            args.model_path,
            "--adapter-path",
            args.adapter_path,
            "--example-mode",
            args.example_mode,
            "--eval-start",
            str(start),
            "--max-eval-examples",
            str(count),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--max-tokens",
            str(args.max_tokens),
            "--predictions-path",
            str(pred_path),
            "--metrics-path",
            str(metrics_path),
        ]
        if args.use_auth_token:
            cmd.append("--use-auth-token")
        if args.no_chat_template:
            cmd.append("--no-chat-template")

        print(f"run chunk start={start} count={count}", flush=True)
        subprocess.run(cmd, check=True)

    rows: list[dict] = []
    for path in sorted(out_dir.glob("pred_*.json")):
        rows.extend(load_rows(path))

    if len(rows) != len(validation):
        raise RuntimeError(f"combined rows={len(rows)} but validation rows={len(validation)}")

    metrics = evaluate(
        raw=[row["raw"] for row in rows],
        gold=[row["gold"] for row in rows],
        pred=[row["pred"] for row in rows],
        info=True,
    )
    final_payload = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "example_mode": args.example_mode,
        "use_chat_template": not args.no_chat_template,
        "validation_rows": len(rows),
        "baseline_lai": metrics[0],
        "accuracy": metrics[1],
        "err": metrics[2],
    }
    final_path = out_dir / "full_metrics.json"
    final_path.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote final_path={final_path}", flush=True)


if __name__ == "__main__":
    main()
