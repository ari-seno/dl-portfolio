Berikut daftar 10 project portfolio deep learning yang sudah direncanakan :

1. NN from Scratch (01-nn-from-scratch) — ✅ Selesai
Membangun neural network murni pakai NumPy (Dense, ReLU, Sigmoid, BinaryCrossEntropy) di dataset make_moons. Yang dipelajari: mekanisme forward/backward propagation dari nol, debugging gradient (menemukan bug double-averaging di losses.py), dan efek hyperparameter (learning rate, hidden dim) terhadap akurasi — hasil akhir 99.8%.

2. PyTorch Reimplementation (02-pytorch-reimpl) — ✅ Selesai
Mengimplementasikan ulang arsitektur Project 1 tapi pakai PyTorch (nn.Linear, autograd, Adam). Yang dipelajari: cara kerja autograd dan optimizer built-in, serta perbandingan langsung "manual vs framework" — hasil test accuracy 100%.

3. MLP Fraud Classifier (03-mlp-fraud-classifier) — ✅ Selesai
Klasifikasi supervised untuk deteksi fraud kartu kredit (dataset Kaggle ~568k baris seimbang). Yang dipelajari: preprocessing data tabular (StandardScaler), arsitektur MLP dengan BatchNorm/Dropout, dan evaluasi model klasifikasi (precision/recall/F1/ROC-AUC) — hasil hampir sempurna (F1 0.9997).

4. Autoencoder Anomaly Detection (04-autoencoder-anomaly-detection) — ✅ Selesai
Deteksi anomali unsupervised di dataset fraud yang sangat imbalance (0.17% fraud), model dilatih hanya dari data normal. Yang dipelajari: konsep reconstruction error, threshold tuning (F1-optimized vs percentile), eksperimen arsitektur (bottleneck/latent dim), dan perbandingan dengan baseline klasik (One-Class SVM, Isolation Forest, LOF) — insight penting bahwa ROC-AUC tinggi tidak selalu berarti F1 bagus.

5. CNN Image Classification (05-cnn-image-classification) — ✅ Selesai
Klasifikasi gambar CIFAR-10 pakai CNN gaya mini-VGG (3 ConvBlock + classifier head). Yang dipelajari: augmentasi data gambar (RandomCrop, RandomHorizontalFlip), arsitektur CNN dari nol, learning rate scheduler (ReduceLROnPlateau), dan checkpointing berdasarkan val accuracy. Status: kode sudah lengkap, tinggal dijalankan lokal (butuh GPU).

6. Transfer Learning — ✅ Selesai
Rencana: memakai model pretrained (misalnya ResNet) untuk klasifikasi gambar, sebagai pembanding terhadap CNN from-scratch di Project 5. Fokus pembelajaran: fine-tuning, feature extraction, dan kapan transfer learning lebih unggul dari training from scratch.

7. LSTM Klasifikasi Teks Bahasa Indonesia — ⏳ Belum dimulai
Rencana: model LSTM untuk klasifikasi teks berbahasa Indonesia. Fokus pembelajaran: pemrosesan data sekuensial/NLP, embedding, dan arsitektur recurrent (LSTM).

8. IndoBERT Fine-tuning — ⏳ Belum dimulai
Rencana: fine-tuning model pretrained IndoBERT untuk tugas NLP Bahasa Indonesia. Fokus pembelajaran: transformer-based NLP, transfer learning di domain teks, dan perbandingan performa dengan LSTM di Project 7.

9. REST API + Docker Deployment — ⏳ Belum dimulai
Rencana: membungkus salah satu model jadi REST API dan deploy pakai Docker. Fokus pembelajaran: MLOps dasar — serving model, containerization, dan deployment pipeline.

10. Coding Agent — ⏳ Belum dimulai
Rencana: mengembangkan Hermes Agent + RAG pipeline dengan tool-calling, integrasi filesystem/terminal/git, opsional LoRA fine-tune di Qwen2.5-Coder 7B. Fokus pembelajaran: agentic AI, tool use, RAG, dan opsional fine-tuning LLM dengan LoRA.