# Data

Raw corpora are not committed to the repository. This document explains how to
obtain the two datasets the pipeline uses and where to place them.

## TinyStories (pre-training)

TinyStories is downloaded automatically from the Hugging Face Hub the first time
data preparation runs, so no manual download is needed. Public home:
[roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

```python
from datasets import load_dataset
load_dataset("roneneldan/TinyStories", split="train")
```

The first download is roughly 1-2 GB and is cached under the standard Hugging
Face cache directory. The pipeline samples the first 10,000 stories, splits them
80/10/10 with seed 42, and fits the tokenizer on the training split only.

## Emotions sentiment sentences (downstream task)

The sentiment classifier uses `emotions_classified.csv`: 16,000 short sentences,
each labeled with a fine-grained emotion and a binary `sentiment` column
(`positive` / `negative`). It corresponds to the train split of the emotion
dataset by Saravia et al. (2018), with each emotion mapped to a binary sentiment
label. Public home:
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion).

Columns:

| Column | Description |
| --- | --- |
| `text` | The sentence |
| `emotion` | Fine-grained emotion label (e.g. joy, sadness, anger) |
| `line_number` | Original line index |
| `sentiment` | Binary label used for training: `positive` or `negative` |

Place the file at:

```
data/emotions_classified.csv
```

The pipeline resolves this path through `qwen3/config.py` (`EMOTIONS_CSV_PATH`).

### Verify

```bash
python -c "import csv; rows=list(csv.DictReader(open('data/emotions_classified.csv', encoding='utf-8'))); print(len(rows), 'rows;', sum(r['sentiment']=='positive' for r in rows), 'positive')"
```

Expected output: `16000 rows; 7238 positive`.

## Splits and seed (single source of truth)

Both datasets use an 80/10/10 train/validation/test split under a fixed seed of
42, applied before any tokenizer fitting or training. These sizes are the canonical
reference for every reported metric; the confidence intervals in the report and
README are bootstrapped over the sentiment test set below.

| Dataset | Total | Train | Validation | Test |
| --- | --- | --- | --- | --- |
| TinyStories (first 10,000) | 10,000 | 8,000 | 1,000 | 1,000 |
| Emotions sentiment | 16,000 | 12,800 | 1,600 | 1,600 |

The sentiment corpus is class-imbalanced overall (7,238 positive, 8,762
negative), which is why the majority-class baseline sits near 56% rather than
50%. The tokenizer is fit on the TinyStories training split only, so both the
TinyStories held-out sets and the entire sentiment corpus are out-of-distribution
for the vocabulary; the tokenizer OOV analysis in the report quantifies this.

## Run

With `data/emotions_classified.csv` in place (TinyStories downloads on first run),
run the pipeline from the repository root:

```bash
python qwen3_pipeline.py          # full pipeline (reproduces the reported results)
python qwen3_pipeline.py --quick  # fast smoke test on a tiny model and data subset
```

## What is and is not committed

- Not committed: `data/emotions_classified.csv`, the TinyStories download,
  regenerated tokenizer vocabulary (`qwen3_tokenizer_vocab.json`), checkpoints,
  and raw logs. These are reproducible and are listed in `.gitignore`.
- Committed: the summary metrics under `artifacts/` and the figures under
  `figures/`, which are sufficient to review the reported results without the
  raw data.
