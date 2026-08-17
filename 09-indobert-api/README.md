# IndoBERT Sentiment API

REST API untuk klasifikasi sentimen teks Bahasa Indonesia menggunakan model **IndoBERT** (fine-tuned `indobert-large-p1`) yang dibangun dengan **FastAPI**. Model membedakan tiga label: `negative`, `neutral`, dan `positive`.

## Ringkasan Project

API ini merupakan layanan inferensi untuk model fine-tuning IndoBERT yang dilatih pada folder `08-indobert-finetuning`. Layanan dibangun dengan FastAPI murni (tanpa framework lain) dan menyediakan endpoint:

- `GET /health` — status kesehatan service dan model
- `GET /metadata` — informasi model (nama, tipe, label, max length)
- `POST /predict` — klasifikasi satu teks
- `POST /predict_batch` — klasifikasi banyak teks sekaligus

## Struktur Folder

```
09-indobert-api/
├── app/
│   ├── __init__.py
│   ├── config.py          # konfigurasi (MODEL_DIR, DEVICE, MAX_LENGTH, LABELS)
│   ├── inference.py       # SentimentModel (load, predict, predict_batch)
│   ├── main.py            # create_app() dan instance FastAPI
│   └── schemas.py         # skema request/response Pydantic
├── bruno/
│   └── collection.json    # koleksi Bruno (health, metadata, predict, predict_batch)
├── tests/                 # unit test & integration test
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Cara Run Lokal

Gunakan interpreter `dl-portfolio` yang memiliki torch dengan dukungan XPU (Intel Arc):

```bash
cd 09-indobert-api
MODEL_DIR=../08-indobert-finetuning/outputs/best_model \
~/.pyenv/versions/3.11.0/envs/dl-portfolio/bin/python -m uvicorn app.main:app --reload
```

Secara default `DEVICE=auto`, sehingga di lokal model akan berjalan di XPU jika tersedia, lalu CUDA, dan terakhir CPU. Model di-load saat startup (lifespan) dari `MODEL_DIR` (default `model/`).

Akses dokumentasi interaktif di http://localhost:8000/docs.

## Cara Run dengan Docker

```bash
cd 09-indobert-api
docker compose up --build
```

File `docker-compose.yml` melakukan bind-mount model dari `../08-indobert-finetuning/outputs/best_model` ke `/app/model` (read-only) dan menetapkan `DEVICE=cpu` karena container tidak punya akses GPU. API berjalan di http://localhost:8000.

## Import Koleksi ke Bruno

1. Buka aplikasi Bruno.
2. Klik **Import** → pilih file `bruno/collection.json`.
3. Koleksi **IndoBERT Sentiment API** akan muncul dengan 4 request siap pakai.

## Daftar Endpoint

### 1. GET `/health`

Cek status service dan model.

Contoh request:

```bash
curl http://localhost:8000/health
```

Contoh respons (200):

```json
{
  "status": "ok",
  "device": "cpu",
  "model_loaded": true
}
```

Jika model belum ter-load, API mengembalikan `503`.

### 2. GET `/metadata`

Informasi model.

Contoh respons (200):

```json
{
  "model_name": "indobert-large-p1",
  "model_type": "BertForSequenceClassification",
  "labels": ["negative", "neutral", "positive"],
  "max_length": 128
}
```

### 3. POST `/predict`

Klasifikasi satu teks. Body JSON: `{"text": "..."}`.

Contoh request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "filmnya bagus sekali"}'
```

Contoh respons (200):

```json
{
  "text": "filmnya bagus sekali",
  "label": "positive",
  "score": 0.9996,
  "probabilities": {
    "negative": 0.0001,
    "neutral": 0.0003,
    "positive": 0.9996
  }
}
```

### 4. POST `/predict_batch`

Klasifikasi banyak teks sekaligus. Body JSON: `{"texts": ["...", "..."]}`.

Contoh request:

```bash
curl -X POST http://localhost:8000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["filmnya bagus sekali", "pelayanannya sangat buruk"]}'
```

Contoh respons (200):

```json
{
  "results": [
    {
      "text": "filmnya bagus sekali",
      "label": "positive",
      "score": 0.9996,
      "probabilities": {
        "negative": 0.0001,
        "neutral": 0.0003,
        "positive": 0.9996
      }
    },
    {
      "text": "pelayanannya sangat buruk",
      "label": "negative",
      "score": 0.9991,
      "probabilities": {
        "negative": 0.9991,
        "neutral": 0.0008,
        "positive": 0.0001
      }
    }
  ]
}
```

## Catatan Model

- **Lokasi model**: di-bind-mount dari `../08-indobert-finetuning/outputs/best_model` (folder hasil fine-tuning task 08).
- **Device**: Docker selalu `DEVICE=cpu`; di lokal memakai XPU (Intel Arc) jika tersedia, fallback ke CUDA/CPU.
- **Output**: setiap teks menghasilkan `label`, `score` (probabilitas label terpilih), dan `probabilities` lengkap untuk ketiga label.