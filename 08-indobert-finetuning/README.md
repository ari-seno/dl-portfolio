# Project 08 — IndoBERT Fine-tuning untuk Sentiment Analysis Bahasa Indonesia (SmSA)

Fine-tuning model transformer pretrained untuk klasifikasi sentimen 3 kelas
(negative / neutral / positive) pada dataset **SmSA** (Indonesian Twitter
sentiment). Fokus pembelajaran: transformer-based NLP, transfer learning di
domain teks, dan perbandingan performa antar model pretrained.

## Dataset

- **Sumber**: SmSA (Indonesian Twitter sentiment) — di-split ke train/valid/test
- **Distribusi** (imbalance, kelas *neutral* minoritas):
  - train: 11.000 (positive 6.416, negative 3.436, neutral 1.148)
  - valid: 1.260
  - test: 500 (positive 208, negative 204, neutral 88)
- Preprocessing: TSV `data/*_preprocess.tsv`, label di-map ke `LABEL2ID`.

## Metrik target

**F1-macro** dipakai sebagai `metric_for_best_model` karena dataset tidak
seimbang dan perbaikan kelas minoritas (*neutral*) penting.

## Hasil eksperimen

### Fase A — Hyperparameter sweep pada `indobert-base-p1`

| Config | Best val F1-macro |
|--------|-------------------|
| **lr 2e-5, tanpa class weights** | **0.9226** ✅ |
| lr 2e-5, class weights | 0.9201 |
| lr 3e-5, tanpa class weights | 0.9200 |
| lr 1e-5, tanpa class weights | 0.9149 |

Class weights (balanced loss) tidak meningkatkan F1-macro. Config terbaik:
**lr 2e-5, tanpa class weights, 5 epochs, early stopping patience 2**.

### Fase B — Perbandingan model kandidat (config terbaik Fase A)

| Model | Best val F1-macro | Test F1-macro |
|-------|-------------------|---------------|
| **`indobert-large-p1`** | **0.9318** ✅ | **0.9101** |
| `xlm-roberta-base` | 0.9238 | 0.9114 |
| `indobert-base-p1` | 0.9226 | 0.8951 |
| `indobert-base-p2` | 0.9189 | 0.8905 |
| `indobertweet-base-p1` | — (gated, butuh auth) | — |

**Model terbaik: `indobenchmark/indobert-large-p1`** — model terbesar memberi
F1-macro tertinggi, konsisten dengan literatur (model berbahasa Indonesia
spesifik > model multilingual untuk sentimen bahasa Indonesia).

### Evaluasi final di test set (`outputs/best_model`)

```
accuracy   : 0.9380
F1-macro   : 0.9101
precision  : 0.9452
recall     : 0.8913
```

Per kelas (F1):
- negative: **0.9691** (recall sempurna 1.0)
- positive: **0.9533**
- neutral : **0.8079** ← kelas minoritas, tetap yang paling sulit

Confusion matrix test:
```
[[204   0   0]   negative: 204 benar, 0 salah
 [ 11  61  16]   neutral : 61 benar, 27 salah (sering jadi positive)
 [  2   2 204]]  positive: 204 benar, 4 salah
```

## Cara pakai

```bash
# Training (default: indobert-base-p1)
PY=~/.pyenv/versions/3.11.0/envs/dl-portfolio/bin/python
$PY -m src.train --model-name indobenchmark/indobert-large-p1 \
    --learning-rate 2e-5 --num-epochs 5 --batch-size 16

# Evaluasi model tersimpan di test set
$PY -m src.evaluate --model-dir outputs/best_model
```

## Hardware & env

- Training dijalankan di **Intel Arc GPU via XPU** (torch `2.13.0+xpu`,
  env `dl-portfolio`), bukan CPU.
- `get_device()` memprioritaskan XPU > CUDA > CPU (`src/train.py`).
- Strategi termal: cooldown sebelum/sesudah training (iGPU berbagi paket
  termal CPU, suhu cepat naik).

## Struktur

- `src/dataset.py` — load TSV, tokenisasi, class weights
- `src/model.py` — build model classification head
- `src/train.py` — training loop (CLI: model, LR, epochs, batch, warmup,
  class weights, early stopping)
- `src/evaluate.py` — evaluasi test set + classification report
- `tests/` — unit test dataset & tokenization