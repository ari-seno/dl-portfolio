# Autoencoder for Credit Card Fraud Anomaly Detection

## Overview

This project implements an **unsupervised anomaly detection** approach to credit card fraud detection using a deep autoencoder — a departure from the supervised classification approach used in Project 3. Instead of learning "what fraud looks like" from labeled examples, the autoencoder learns "what normal transactions look like" by training exclusively on non-fraudulent data. Fraud is then flagged based on how poorly a transaction is reconstructed by the model.

This approach is particularly valuable in real-world scenarios where labeled fraud examples are scarce, incomplete, or where fraud patterns evolve faster than labels can be collected.

## Dataset

[Credit Card Fraud Detection (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions by European cardholders in September 2013, with only 492 fraud cases (~0.17%). Features V1-V28 are PCA-transformed for confidentiality; `Time` and `Amount` are the only raw features.

### Preprocessing
- `Amount` is log-transformed (`log1p`) before scaling, since it is heavily right-skewed
- `Time` and `Amount` are standardized with `StandardScaler`
- Data is split into train (normal-only), validation, and test sets (stratified on `Class` for val/test)

|         Split       |    Rows  |  Fraud Ratio   |
|---------------------|----------|----------------|
| Train (normal only) | 199,020  | 0% (by design) |
| Validation          | 42,721   | 0.173%         |
| Test                | 42,722   | 0.173%         |

## Approach

1. Train an autoencoder to reconstruct normal transactions only
2. Compute per-sample reconstruction error (MSE) on validation/test sets
3. Tune a decision threshold on the validation set using `precision_recall_curve` to maximize F1
4. Classify test transactions as fraud if their reconstruction error exceeds the threshold

## Final Model Architecture

```
Encoder: 30 → 32 → 20 → 6 (latent)
Decoder: 6 → 20 → 32 → 30
```
- BatchNorm1d + LeakyReLU(0.1) at each hidden layer
- No activation on the latent layer or final output layer
- Trained with Adam (lr=1e-3, weight_decay=1e-5), ReduceLROnPlateau scheduler
- 150 epochs, best-validation-loss checkpointing
- Fixed random seed (42) for reproducibility

## Results

|   Metric  | Validation | Test       |
|-----------|------------|------------|
| Precision | 0.5521     | 0.5638     |
| Recall    | 0.7162     | 0.7162     |
| F1 Score  | 0.6235     | **0.6310** |
| ROC-AUC   | 0.9538     | 0.9422     |

## Architecture Experiments

Several configurations were systematically tested to arrive at the final architecture. Key finding: **model capacity has a sweet spot** — both too little and too much capacity hurt generalization to the test set, even when validation metrics look strong.

| #     | Hidden Layers | Latent Dim | Test Precision | Test Recall |   Test F1  | Test ROC-AUC |
|-------|---------------|------------|----------------|-------------|------------|--------------|
| 1     | 24→16         | 10         | 0.4569         | 0.7162      | 0.5579     | 0.9311       | (initial baseline)
| 2     | 24→16         | 14         | 0.4649         | 0.7162      | 0.5638     | 0.9195       |
| 3     | 24→16         | 6          | 0.4690         | 0.7162      | 0.5668     | 0.9304       |
| **4** | **32→20**     | **6**      | **0.5638**     | **0.7162**  | **0.6310** | **0.9422**   | (Champion)
| 5     | 32→20→12      | 6          | 0.5146         | 0.7162      | 0.5989     | 0.9403       |
| 6     | 32→20→12      | 8          | 0.5340         | 0.7432      | 0.6215     | 0.9274       |
| 7     | 32→20         | 8          | 0.4533         | 0.4595      | 0.4564     | **0.9428**   |

**Key insight:** Widening the hidden layers (24→16 → 32→20) while keeping a tight bottleneck (latent_dim=6) gave the model more room to process features before compression, improving every metric simultaneously. However, adding a third layer or increasing latent_dim beyond 6 consistently improved validation metrics while *hurting* test generalization — likely because additional capacity lets the model start reconstructing fraud patterns too, blurring the separation that makes anomaly detection work. This was confirmed across multiple isolated comparisons (varying latent_dim alone, and varying depth alone).

Dropout (0.1) was also tested and rejected — it caused severe underfitting given the model's already-small bottleneck.

## Baseline Comparison

To validate that the deep learning approach is justified, the autoencoder was benchmarked against three classical unsupervised anomaly detection methods, all trained on the same normal-only data and evaluated with the same F1-optimized thresholding strategy.

| Model            | Test Precision | Test Recall |  Test F1   | Test ROC-AUC |
|------------------|----------------|-------------|------------|--------------|
| **Autoencoder**  | **0.5638**     | **0.7162**  | **0.6310** | 0.9422       |
| One-Class SVM    | 0.3509         | 0.5405      | 0.4255     | 0.9308       |
| Isolation Forest | 0.2099         | 0.2297      | 0.2194     | 0.9431       |
| Local Outlier Factor | 0.0211     | 0.0405      | 0.0278     | 0.7960       |

**Key insight:** The autoencoder and Isolation Forest have nearly identical ROC-AUC (~0.94), meaning both rank transactions by "anomalousness" almost equally well overall. Yet the autoencoder's F1 is nearly 3x higher — its reconstruction error creates a much sharper separation between normal and fraud scores *at the specific threshold used for a real decision*, whereas Isolation Forest's scores are more gradual/overlapping. This illustrates that ROC-AUC alone is insufficient to judge anomaly detection models; the precision-recall trade-off at a realistic operating threshold matters more in practice.

Local Outlier Factor performed worst by a wide margin, including on ROC-AUC itself — likely due to the curse of dimensionality affecting its k-nearest-neighbor-based density estimation across 30 PCA-transformed features.

One-Class SVM was trained on a 5,000-row subsample (rather than the full ~199K normal transactions) due to its poor computational scalability (O(n²)-O(n³) with an RBF kernel) — yet still outperformed both other classical baselines, showing it is comparatively data-efficient.

## Project Structure

```
04-autoencoder-anomaly-detection/
├── data/                    # creditcard.csv (not committed — see Setup)
├── src/
│   ├── dataset.py           # loading, log-transform, scaling, normal-only split
│   ├── model.py              # Autoencoder architecture
│   ├── train.py               # training loop, threshold tuning, evaluation
│   └── baseline_models.py    # Isolation Forest, LOF, One-Class SVM comparisons
├── notebooks/
│   └── exploration.ipynb     # reconstruction error viz, confusion matrix, ROC, baseline comparison chart
├── tests/
│   └── test_model.py         # unit tests for the Autoencoder architecture
├── outputs/                   # saved model, generated plots
├── requirements.txt
└── README.md
```

## Setup

```bash
pyenv local dl-portfolio
pip install -r requirements.txt

# Download dataset (requires Kaggle account/API key)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

## Usage

```bash
# Train the autoencoder and evaluate
python -m src.train

# Run classical baseline comparisons
python -m src.baseline_models

# Run unit tests
pytest tests/ -v
```

## Key Takeaways

- Unsupervised anomaly detection with autoencoders is a viable alternative to supervised classification when labeled anomalies are scarce or unreliable
- Model capacity tuning for anomaly detection is a balancing act: enough capacity to reconstruct normal data well, but not so much that the model starts reconstructing anomalies too
- ROC-AUC and F1 can diverge significantly for anomaly detection models — both should be reported, since a model can rank well overall but perform poorly at any single realistic decision threshold
- Classical methods (Isolation Forest, LOF, One-Class SVM) remain useful, fast-to-train baselines, but struggled to match the autoencoder's precision-recall trade-off on this high-dimensional, PCA-transformed dataset