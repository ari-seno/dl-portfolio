02 - PyTorch Reimplementation

Reimplementasi arsitektur neural network dari Project 1 (01-nn-from-scratch) menggunakan PyTorch, pada dataset yang sama (make_moons), untuk membandingkan hasil "manual NumPy" dengan implementasi framework secara apples-to-apples.

Tujuan

Project 1 membangun neural network dari nol (forward/backward pass manual, gradient descent manual) untuk memahami mekanisme di balik layar. Project 2 membangun ulang arsitektur yang sama menggunakan nn.Linear, nn.ReLU, nn.Sigmoid, nn.BCELoss, autograd, dan optimizer bawaan PyTorch — untuk memvalidasi pemahaman manual sekaligus membiasakan diri dengan workflow training ala framework modern.

Arsitektur
Input (2) → Linear(2, 32) → ReLU → Linear(32, 1) → Sigmoid → BCELoss

Arsitektur, hidden dimension, dan hyperparameter mengikuti konfigurasi terbaik dari Project 1 (hidden_dim=32) agar hasil dapat dibandingkan langsung.

Struktur Folder
02-pytorch-reimpl/
├── data/
├── src/
│   ├── __init__.py
│   ├── model.py          # Definisi NeuralNetwork dengan nn.Module
│   └── train.py           # Loop training, evaluasi, train/val/test split
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   └── test_model.py
├── outputs/
├── requirements.txt
├── README.md
└── .gitignore

Berbeda dari Project 1, tidak ada layers.py, activations.py, atau losses.py terpisah karena semua komponen tersebut sudah disediakan langsung oleh PyTorch (torch.nn, torch.nn.functional, autograd).

Setup
bash
pyenv local dl-portfolio
pip install -r requirements.txt
Menjalankan Training
bash
python -m src.train
Menjalankan Test
bash
pytest tests/test_model.py -v
Hasil
Epoch	Loss	Train Acc	Val Acc
1	0.6592	0.7675	0.7400
500	0.1043	0.9700	0.9800
1000	0.0442	0.9900	0.9900
1500	0.0294	0.9950	0.9900
2000	0.0226	0.9950	1.0000
2500	0.0186	0.9950	1.0000
3000	0.0159	0.9975	1.0000
3500	0.0138	0.9975	1.0000
4000	0.0121	0.9975	1.0000

Final Train Accuracy: 0.9975 Final Test Accuracy: 1.0000

Perbandingan dengan Project 1 (NumPy manual)
Implementasi	Final Train Acc	Final Test/Val Acc
Project 1 (NumPy)	0.9980	-
Project 2 (PyTorch)	0.9975	1.0000

Hasil kedua implementasi konsisten, mengonfirmasi bahwa implementasi manual di Project 1 (forward/backward pass, gradient descent) sudah benar secara matematis — PyTorch hanya menggantikan perhitungan gradien manual dengan autograd tanpa mengubah hasil akhir secara signifikan.

Insight
Autograd PyTorch menghilangkan kebutuhan menulis backward() manual per layer, cukup mendefinisikan forward pass dan memanggil loss.backward().
Hasil akhir yang hampir identik dengan Project 1 menjadi validasi silang bahwa pemahaman mekanisme backpropagation manual sudah solid sebelum berpindah ke tools yang lebih tinggi level.
Val Acc mencapai 1.0000 lebih cepat (sejak epoch 2000) dibanding Train Acc, konsisten dengan sifat dataset make_moons yang relatif mudah dipisahkan pada noise rendah.