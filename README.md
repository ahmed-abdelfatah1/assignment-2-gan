# DSAI 490 Assignment 2 — Conditional Date Generator

Generates a date `d-m-yyyy` (range 1-1-1800 to 31-12-2200) that satisfies four
input conditions: `[day_of_week] [month] [leap] [decade]`. Four models are
implemented in PyTorch, with hand-written training loops:

| model         | output                              | role                       |
|---------------|--------------------------------------|----------------------------|
| cGAN          | 31\|10 softmax (day, year last digit) | mandatory                  |
| cVAE          | 31\|10 softmax (day, year last digit) | from the course            |
| cLSTM         | char sequence `d-m-yyyy`             | outside the course         |
| cTransformer  | char sequence `d-m-yyyy`             | outside the course         |

Conditions encode as a 62-dim one-hot vector (7 + 12 + 2 + 41). The cGAN/cVAE
only have to predict the day and year-last-digit because `[month]` and
`[decade]` are already given.

## Train

Local (CPU):

```
python -m pytest tests/                                  # unit tests
python model/train.py --model all                        # all four (uses GPU if visible)
python model/train.py --model ctransformer --epochs 2    # smoke
```

Colab (T4 / any GPU): open `notebooks/train_colab.ipynb` and run top-to-bottom.
The notebook clones the repo, trains all four models, evaluates them, and zips
`model/weights/` + `report/` into `artifacts.zip` for download.

## Predict (assignment-spec CLI)

```
python model/predict.py -i data/example_input.txt -o predictions.txt
```

By default this runs all four models, writes four sibling files
(`predictions_cgan.txt`, `predictions_cvae.txt`, `predictions_clstm.txt`,
`predictions_ctransformer.txt`), and writes `predictions.txt` itself using the
transformer's outputs. Each input row is sampled up to **K=20** times and the
first sample that satisfies the four conditions is accepted; if all 20 fail the
last sample is emitted (and the line is counted as a failure in the summary).

## Reproduce

```
conda env create -f environment.yml
conda activate dsai490-a2
python model/train.py --model all
python model/evaluate.py
```

All randomness is seeded from `config.SEED = 42` (Python `random`, NumPy, and
PyTorch CPU + CUDA). The 80/10/10 train/val/test split is deterministic.
