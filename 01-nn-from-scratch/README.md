# 01 — Neural Network from Scratch (NumPy)

Implementasi neural network dari nol menggunakan NumPy murni — tanpa framework seperti PyTorch atau TensorFlow. Project ini adalah fondasi dari roadmap 10-project deep learning portfolio menuju AI Engineer.

## Tujuan

Membangun pemahaman mendalam tentang cara kerja internal neural network:
- Forward propagation
- Backpropagation (chain rule) secara manual
- Gradient descent
- Loss function dan optimisasi

## Arsitektur

Input (2 fitur) → Dense(32) → ReLU → Dense(1) → Sigmoid → Output (probabilitas)


Binary classifier sederhana untuk memisahkan dua kelas non-linear pada dataset `make_moons`.

## Struktur Project

01-nn-from-scratch/
├── data/
├── src/
│ ├── layers.py # Dense layer (forward & backward)
│ ├── activations.py # ReLU, Sigmoid, Softmax
│ ├── losses.py # Binary Cross-Entropy
│ ├── model.py # NeuralNetwork class
│ └── train.py # Training loop + data loading (train/test split)
├── notebooks/
│ └── exploration.ipynb # Visualisasi loss curve & decision boundary
├── tests/
│ └── test_layers.py # Unit test + numerical gradient checking
├── outputs/ # Hasil plot (loss curve, decision boundary)
├── requirements.txt
└── README.md


## Setup

```bash
pyenv local dl-portfolio
pip install -r requirements.txt
```

## Menjalankan Training

```bash
python -m src.train
```

## Menjalankan Test

```bash
pytest tests/test_layers.py -v
```

## Hasil

| Konfigurasi | Accuracy |
|---|---|
| hidden_dim=16, lr=0.1, epochs=1000 | 88.4% |
| hidden_dim=16, lr=0.2, epochs=3000 | 93.2% |
| hidden_dim=32, lr=0.2, epochs=1000 | 93.0% |
| **hidden_dim=32, lr=0.3, epochs=4000, noise=0.15** | **99.8%** |

Hyperparameter final: `hidden_dim=32`, `learning_rate=0.3`, `epochs=4000`, `noise=0.15`.

## Debugging Notes

Ditemukan bug pada implementasi awal `BinaryCrossEntropy.backward()`: gradient dibagi jumlah sample (`m`) dua kali — sekali di loss, sekali lagi di `Dense.backward()` — menyebabkan gradient efektif mendekati nol dan model gagal belajar (accuracy stuck di ~49%, setara tebakan acak). Diperbaiki dengan menghapus pembagian `/m` yang redundan di `losses.py`, sehingga averaging hanya terjadi sekali.

## Insight

- Menurunkan noise dataset dari 0.2 ke 0.15 memberi dampak lebih besar terhadap accuracy dibanding menambah `hidden_dim` — karena bottleneck sebenarnya adalah overlap kelas pada data, bukan kapasitas network.
- Numerical gradient checking (finite-difference) di `tests/test_layers.py` adalah teknik verifikasi yang seharusnya bisa mendeteksi bug backward pass sejak awal, sebelum training dijalankan.

## Next Steps

- Konfirmasi hasil train/test split (Train Acc vs Test Acc) untuk laporan generalization accuracy yang lebih kredibel dibanding training accuracy saja.
- Lanjut ke **Project 2: PyTorch Reimplementation**.