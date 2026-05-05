# MultiLexNorm 2026 Demo
This repository provides a demo for the MultiLexNorm shared task.
It demonstrates how to download the dataset, run a simple baseline model (MFR), and evaluate normalization results.


- [**Full code is available here**](demo.ipynb).
- The datasets will be available at [development phase](https://huggingface.co/datasets/weerayut/multilexnorm2026-dev-pub) and [final phase](https://huggingface.co/datasets/weerayut/multilexnorm2026-dev-pub).
- Example MFR submission outputs: `outputs/submission_dev.zip` and `outputs/submission_full.zip`

## Assignment extensions

This project keeps the provided baseline structure and extends it:

- `utils.py` still contains the shared MFR, evaluation, and zip helpers from the baseline.
- `mfr_baseline.py` is a script version of the original per-language MFR demo.
- `byt5_model.py` is the proposed model; it uses the same dataset splits, token-level prediction format, and `utils.evaluate()` metric.
- `model.py` is a lightweight exploratory MFR+ variant kept for comparison, not the main proposed model.

For report comparison, use `mfr_baseline.py` as the provided baseline and `byt5_model.py` as the fine-tuned model.

## Set up the environment
```bash
# Create an environment and install packages
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Load data

```python
from datasets import load_dataset

pub_data = load_dataset("weerayut/multilexnorm2026-pub")

# Select a language
lang = "en"
en_train = pub_data["train"].filter(lambda x: x["lang"] == lang)
en_val = pub_data["validation"].filter(lambda x: x["lang"] == lang)
```


## Inference
```python
import pandas as pd
from utils import counting, mfr

# Smoke test the baseline
counts = counting(en_train)
mfr(['bcause', 'u', 'r', 'funny'], counts)

# Inference
ds = pd.DataFrame(en_val)
ds['pred'] = ds['raw'].apply(lambda x: mfr(x, counts))
```

## Evaluation
```python
from utils import evaluate

evaluate(
    raw=ds['raw'].tolist(),    # list[list[str]]
    gold=ds['norm'].tolist(),  # list[list[str]]
    pred=ds['pred'].tolist()   # list[list[str]]
)
```
Output:
```txt
Baseline acc.(LAI): 93.10
Accuracy:           97.37
ERR:                61.93
```

## Lightweight model

`model.py` contains the assignment model used for reproducible runs. It keeps the
provided MFR baseline as the main behavior, then adds exact global and lowercase
fallback tables so unseen language-specific tokens can still reuse evidence from the
training data.

```bash
# Evaluate on the validation split
python model.py --eval-only

# Generate test predictions and zip them for submission
python model.py \
  --dataset weerayut/multilexnorm2026-dev-pub \
  --output-dir outputs/submission_dev
```

If the Hugging Face dataset requires authentication, first run
`huggingface-cli login`, then add `--use-auth-token` to either command.

## Exact MFR baseline

`mfr_baseline.py` follows the provided notebook baseline as a reproducible script.

```bash
python mfr_baseline.py --eval-only --use-auth-token
```

## ByT5 model

`byt5_model.py` fine-tunes `google/byt5-small` as a byte-level token
normalizer. Each training example maps one raw token plus its language tag to
one normalized token, which preserves the required prediction length even when
the normalized form contains spaces.

```bash
# Fast pipeline test with a tiny random ByT5 checkpoint
python byt5_model.py \
  --smoke-test \
  --model-name local-tiny-random-byt5 \
  --model-dir models/byt5_smoke

# Fine-tune the real base model on a bounded local subset
python byt5_model.py \
  --train \
  --use-auth-token \
  --model-name google/byt5-small \
  --model-dir models/byt5 \
  --max-train-examples 5000 \
  --epochs 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 8

# Evaluate a trained checkpoint
python byt5_model.py --eval-only --use-auth-token --model-dir models/byt5 --batch-size 128
```

During evaluation and test prediction, ByT5 predictions are globally batched
across all sentence tokens and then reconstructed into the original sentence
format. On a T4 GPU, increase `--batch-size` until memory is near full or CUDA
runs out of memory.
