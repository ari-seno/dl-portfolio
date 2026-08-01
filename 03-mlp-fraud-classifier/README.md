# 03 — MLP Fraud Classifier

MLP (Multi-Layer Perceptron) untuk klasifikasi fraud transaksi kartu kredit menggunakan PyTorch, dilatih pada dataset balanced.

## Dataset

**Credit Card Fraud Detection Dataset 2023** (Kaggle, by nelgiriyewithana)
- 568,630 transaksi
- 31 kolom: `id`, `V1`–`V28` (fitur PCA), `Amount`, `Class` (target)
- Distribusi kelas: **50/50 balanced** (284,315 fraud vs 284,315 non-fraud)

## Arsitektur Model

Input (29 fitur)
→ Linear(29, 128) → BatchNorm1d → ReLU → Dropout(0.3)
→ Linear(128, 64) → BatchNorm1d → ReLU → Dropout(0.3)
→ Linear(64, 32) → BatchNorm1d → ReLU → Dropout(0.2)
→ Linear(32, 1) → Sigmoid


## Struktur Folder

03-mlp-fraud-classifier/
├── data/ # creditcard_2023.csv (gitignored)
├── src/
│ ├── init.py
│ ├── dataset.py # load_data: preprocessing + stratified split
│ ├── model.py # FraudMLP
│ └── train.py # training loop + evaluasi + checkpoint
├── notebooks/
│ └── exploration.ipynb # confusion matrix, ROC curve
├── tests/
│ └── test_model.py
├── outputs/ # model.pt, confusion_matrix.png, roc_curve.png
├── requirements.txt
├── README.md
└── .gitignore


## Setup

```bash
pyenv activate dl-portfolio
pip install -r requirements.txt
```

## Menjalankan

```bash
# Cek preprocessing & split
python -m src.dataset

# Cek arsitektur model
python -m src.model

# Training (menyimpan checkpoint ke outputs/model.pt)
python -m src.train

# Unit test
pytest tests/test_model.py -v
```

## Preprocessing

- Kolom `id` di-drop (bukan fitur)
- `V1`–`V28` sudah PCA-scaled dari sananya, tidak diubah
- `Amount` di-scale dengan `StandardScaler`, **di-fit hanya pada train set** untuk mencegah data leakage
- Split stratified 70% train / 10% val / 20% test, menjaga rasio 50/50 di setiap subset

## Hasil Training

Hyperparameter: batch_size=512, epochs=50, optimizer=Adam(lr=1e-3), loss=BCELoss

| Epoch | Val Loss | Val Acc | Precision | Recall | F1 | ROC-AUC |
|-------|----------|---------|-----------|--------|-----|---------|
| 1 | 0.0175 | 0.9947 | 0.9940 | 0.9953 | 0.9947 | 0.9997 |
| 10 | 0.0042 | 0.9993 | 0.9987 | 1.0000 | 0.9993 | 1.0000 |
| 30 | 0.0033 | 0.9996 | 0.9992 | 1.0000 | 0.9996 | 1.0000 |
| 50 | 0.0029 | 0.9997 | 0.9993 | 1.0000 | 0.9997 | 1.0000 |

**Evaluasi Test Set (final):**

| Metrik | Nilai |
|--------|-------|
| Accuracy | 0.9997 |
| Precision | 0.9994 |
| Recall | 1.0000 |
| F1-score | 0.9997 |
| ROC-AUC | 1.0000 |

**Confusion Matrix (test set, 113,726 sampel):**

|  | Predicted Non-Fraud | Predicted Fraud |
|---|---|---|
| **Actual Non-Fraud** | 56,806 | 57 (FP) |
| **Actual Fraud** | 0 (FN) | 56,863 |

## Insight

- Model mencapai recall 100% (tidak ada false negative sama sekali) dan precision 99.94% pada test set.
- Skor setinggi ini **konsisten dengan karakteristik dataset**: "Credit Card Fraud Detection Dataset 2023" adalah dataset yang digenerate ulang dan sangat separable secara fitur (V1-V28 sudah sangat diskriminatif antar kelas fraud/non-fraud), berbeda dari dataset ULB asli yang jauh lebih noisy dan realistis.
- Dataset ini balanced dari sananya, sehingga project ini tidak butuh teknik penanganan imbalance (class weighting, SMOTE, threshold tuning) yang biasanya jadi fokus utama fraud detection dunia nyata.
- BatchNorm + Dropout terbukti efektif menjaga training stabil meski model belum menunjukkan tanda overfitting hingga epoch 50 (val metrics terus stabil/naik).

## Perbandingan dengan Project 1 & 2

| | Project 1 (NumPy) | Project 2 (PyTorch) | Project 3 (MLP Fraud) |
|---|---|---|---|
| Dataset | make_moons (500 sampel) | make_moons (500 sampel) | Credit Card Fraud 2023 (568,630 sampel) |
| Fitur | 2 | 2 | 29 |
| Arsitektur | 1 hidden layer (manual) | 1 hidden layer (nn.Module) | 3 hidden layer + BatchNorm + Dropout |
| Final Accuracy | 0.9980 | 1.0000 (test) | 0.9997 (test) |

## Next Steps

- Project 4: Autoencoder untuk anomaly detection (pendekatan unsupervised untuk fraud detection)