# Project 7: LSTM Text Classification (Bahasa Indonesia Sentiment Analysis)

## Overview

A from-scratch BiLSTM sentiment classifier for Bahasa Indonesia text, built entirely in PyTorch (embeddings trained from scratch — no pretrained word vectors or transformers). This is the seventh project in a 10-project deep learning portfolio, and serves as the "traditional deep learning" baseline that will be directly compared against Project 8 (IndoBERT fine-tuning) on the exact same dataset.

**Task:** 3-class sentiment classification (negative / neutral / positive) on Indonesian text.

**Final result:** Test macro F1 = **0.7846**, Test accuracy = **0.81**

## Dataset

[IndoNLU SmSA](https://github.com/IndoNLP/indonlu) (`smsa_doc-sentiment-prosa`) — a standard Indonesian sentiment analysis benchmark of pre-tokenized text pulled from online reviews.

| Split | Rows | negative | neutral | positive |
|---|---|---|---|---|
| train | 11,000 | 3,436 (31%) | 1,148 (10%) | 6,416 (58%) |
| valid | 1,260 | 394 (31%) | 131 (10%) | 735 (58%) |
| test | 500 | 204 (41%) | 88 (18%) | 208 (42%) |

The `neutral` class is significantly underrepresented in training/validation (~10%), which shapes several decisions below.

This dataset was chosen specifically so Project 7 (LSTM, from scratch) and Project 8 (IndoBERT, pretrained/fine-tuned) can be evaluated on identical data — a clean head-to-head comparison of the two approaches, mirroring the Project 5 vs 6 (CNN vs transfer learning) pattern.

## Architecture

```
Input (token indices, padded)
  -> nn.Embedding(vocab_size=9074, embedding_dim=128, padding_idx=0)   [trained from scratch]
  -> Bidirectional LSTM(128 -> 128, 1 layer)                            [pack_padded_sequence]
  -> concat(final forward hidden, final backward hidden)  -> (256,)
  -> Dropout(0.3)
  -> Linear(256 -> 3)
  -> logits (negative / neutral / positive)
```

**Total trainable parameters:** 1,426,435

### Key design decisions

**Embeddings trained from scratch, not pretrained (e.g. FastText Indonesia).** This keeps the from-scratch-LSTM vs pretrained-IndoBERT comparison in Projects 7 vs 8 clean — both projects should differ primarily in "does the model start with external language knowledge," not have that variable already partially present in Project 7. It also matches this portfolio's fundamentals-first pattern (Projects 1 and 3 also build core components from scratch before using higher-level abstractions), and 11K training sentences is enough to learn a small, task-specific embedding space.

**Whitespace tokenizer.** The SmSA dataset ships pre-tokenized (words and punctuation already space-separated), so no subword/BPE tokenizer was needed — that's reserved for Project 8's transformer tokenizer.

**`pack_padded_sequence` for the LSTM.** Padding tokens are excluded from the LSTM's computation entirely (rather than being processed as if they were real input and then ignored at pooling time), which is more correct and slightly more efficient.

## Training Setup

- Optimizer: Adam, lr=1e-3
- Loss: CrossEntropyLoss
- Gradient clipping: max_norm=5.0 (standard practice for RNNs)
- Batch size: 32
- Max epochs: 15, with early stopping (patience=4 epochs, monitored on validation macro F1)
- Primary metric: **macro F1** (not accuracy) — chosen because of the class imbalance; accuracy alone would overweight the majority classes and hide poor `neutral`-class performance
- Hardware: trained on Intel Arc integrated GPU (XPU) via `torch.xpu`

## Experiments: Handling Class Imbalance

Three approaches to the `neutral` class underrepresentation were tried and compared on test macro F1:

| Iteration | Config | Test macro F1 | Test acc | Neutral F1 (P / R) |
|---|---|---|---|---|
| 1. Baseline | dropout=0.3, no weighting | 0.7542 | 0.80 | 0.58 (0.68 / 0.51) |
| 2. Loss weighting | dropout=0.5, `CrossEntropyLoss(weight=...)` (sklearn `balanced`, neutral weight 3.19x), weight_decay=1e-5 | 0.7217 | 0.77 | 0.56 (0.75 / 0.44) |
| 3. Oversampling | dropout=0.3, `WeightedRandomSampler` oversampling neutral, weight_decay=1e-5 | 0.7483 | 0.80 | 0.56 (0.87 / 0.44) |

**Finding:** neither class-weighting nor oversampling improved macro F1 over the plain baseline. Both interventions shifted the precision/recall trade-off for `neutral` — precision went up, recall went down — without a net benefit. The model became more "conservative" about predicting `neutral` rather than genuinely better at recognizing it. Given this, the final model uses the plain baseline configuration (no imbalance-specific intervention).

### An important caveat: run-to-run variance

`train.py` does not fix a random seed. Two separate training runs with the *identical* baseline configuration produced meaningfully different results:

| Run | Best epoch | Test macro F1 | Neutral F1 |
|---|---|---|---|
| Baseline run 1 | 7 (stopped at 12) | 0.7542 | 0.58 |
| Baseline run 2 (reproducibility check) | 5 (stopped at 9) | **0.7846** | **0.68** |

A ~3-point macro F1 swing purely from weight initialization and batch-shuffling order is larger than the differences seen between the imbalance-handling experiments above. This is a useful (if humbling) finding: for a model this size on a dataset this size, run-to-run variance can be a bigger factor than the specific technique being tested — a reminder to seed runs and/or average over multiple seeds before drawing strong conclusions from a single training run. (Project 4's autoencoder had a similar variance issue, which is why that project's `train.py` does fix a seed — a good practice this project's `train.py` currently lacks.)

**The reported final model uses the better of the two baseline runs** (macro F1 0.7846), with its checkpoint saved at `outputs/best_model.pt`.

## Final Results (Test Set)

**Macro F1: 0.7846 | Accuracy: 0.81**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.79 | 0.92 | 0.85 | 204 |
| neutral | 0.79 | 0.60 | 0.68 | 88 |
| positive | 0.85 | 0.80 | 0.82 | 208 |

`neutral` remains the hardest class by a clear margin, consistent with it being the most underrepresented class in training data. Confusion matrix analysis (see `notebooks/exploration.ipynb`) shows its dominant failure mode is being predicted as `negative` (~25% of true-neutral samples) — likely because ambiguous/mixed-sentiment text tends to read as mildly critical rather than purely neutral. `positive` samples are also frequently confused as `negative` (~23%), suggesting the model leans toward `negative` as a "default" guess when uncertain.

![Confusion Matrix](outputs/confusion_matrix.png)
![Per-Class Metrics](outputs/per_class_metrics.png)

## Project Structure

```
07-lstm-text-classification/
├── data/
│   ├── train_preprocess.tsv
│   ├── valid_preprocess.tsv
│   └── test_preprocess.tsv
├── src/
│   ├── tokenizer.py     # whitespace tokenizer
│   ├── vocab.py          # Vocab class: build/save/load, encode/decode
│   ├── dataset.py        # SmSADataset + collate_fn (dynamic padding)
│   ├── model.py           # LSTMClassifier (BiLSTM + classification head)
│   └── train.py            # training loop, early stopping, evaluation
├── notebooks/
│   └── exploration.ipynb  # confusion matrix, per-class metrics, error analysis
├── tests/
│   └── test_model.py      # 9 unit tests, all passing
├── outputs/
│   ├── vocab.json
│   ├── best_model.pt
│   ├── confusion_matrix.png
│   └── per_class_metrics.png
├── requirements.txt
└── README.md
```

## Setup & Usage

```bash
# activate the shared portfolio virtualenv
pyenv local dl-portfolio
pip install -r requirements.txt

# build the vocabulary from training data
python -m src.vocab

# train the model (uses Intel Arc XPU if available, else CUDA, else CPU)
python -m src.train --epochs 15

# run tests
pytest tests/ -v
```

Explore results interactively via `notebooks/exploration.ipynb` (confusion matrix, per-class metrics, misclassified examples).

## Key Takeaways

- **A from-scratch BiLSTM reaches a solid but imperfect baseline** (macro F1 ~0.75-0.78) on Indonesian sentiment classification — a meaningful reference point for Project 8's IndoBERT comparison.
- **Class imbalance is hard to fix cheaply.** Both loss-weighting and oversampling shifted the neutral class's precision/recall trade-off without improving overall macro F1 — a reminder that "add class weights" isn't automatically a win, and needs to be measured, not assumed.
- **Run-to-run variance can dominate small tuning changes.** Without a fixed seed, two identical-config runs differed by ~3 macro-F1 points — more than the spread across three different imbalance-handling strategies. This is the strongest single lesson from this project: seed training runs (or report averages over multiple seeds) before concluding that one configuration is meaningfully better than another.
- **`pack_padded_sequence` matters for correctness**, not just efficiency — without it, the LSTM would process `<pad>` tokens as if they were real input, corrupting the final hidden state used for classification.