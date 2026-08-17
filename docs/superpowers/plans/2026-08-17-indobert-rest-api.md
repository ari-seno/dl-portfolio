# Project 09 — IndoBERT Sentiment REST API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membungkus model IndoBERT sentiment (project 08) sebagai REST API FastAPI + Docker, dengan endpoint health/predict/batch/metadata, testing pytest, dan koleksi Bruno.

**Architecture:** Aplikasi FastAPI memuat model saat startup (lifespan hook), inference lewat wrapper `SentimentModel` (XPU > CUDA > CPU). Docker image berisi kode saja; model 1.3GB di-bind-mount dari project 08 via volume. Uji API memakai TestClient dengan model fiktif (cepat, deterministik).

**Tech Stack:** FastAPI, Uvicorn, PyTorch, HuggingFace Transformers, Pydantic, pytest, Docker, Docker Compose.

**Spec:** Desain yang disetujui di chat (folder `09-indobert-api/`, endpoint health/predict/predict_batch/metadata, docs OpenAPI + koleksi Bruno). Tidak ada file spec terpisah — desain tercatat dalam percakapan brainstorming.

## Global Constraints

- Env Python: `dl-portfolio` (`~/.pyenv/versions/3.11.0/envs/dl-portfolio/bin/python`).
- Device prioritas: XPU > CUDA > CPU (konsisten `src/train.py` p08); override via env `DEVICE`.
- Load model EKSPLISIT `num_labels=3, id2label, label2id` (config punya `_num_labels: 5` yang salah → tidak dipakai).
- Model di-mount (ro) dari `../08-indobert-finetuning/outputs/best_model`, tidak di-copy ke image.
- Tidak ada komentar berlebihan; Bahasa Indonesia di README.
- Branch kerja: `09-indobert-rest-api` (konvensi repo: branch per project, merge via PR).
- FastAPI murni — TANPA LangChain (keputusan user).

---

## File Structure

```
09-indobert-api/
├── app/
│   ├── __init__.py
│   ├── config.py      # env vars: MODEL_DIR, DEVICE, MAX_LENGTH, LABELS
│   ├── schemas.py     # Pydantic request/response models
│   ├── inference.py   # class SentimentModel (load/predict/predict_batch)
│   └── main.py        # FastAPI app, lifespan, 4 endpoints
├── tests/
│   ├── test_inference.py  # unit test wrapper (stub torch model)
│   └── test_api.py        # TestClient endpoint tests (stub model)
├── bruno/
│   └── collection.json    # koleksi Bruno
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .gitignore
├── .python-version        # 3.11.0 → auto-pakai env dl-portfolio
└── README.md
```

## Task 1: Scaffolding & config

**Files:**
- Create: `09-indobert-api/.gitignore`, `.python-version`, `requirements.txt`, `app/__init__.py`, `app/config.py`

**Interfaces:**
- Produces: `app.config` dengan `MODEL_DIR`, `DEVICE`, `MAX_LENGTH`, `LABELS`, `LABEL2ID`, `ID2LABEL`.

- [ ] **Step 1**: Buat file dasar
  - `.python-version` berisi `3.11.0`
  - `.gitignore`: `__pycache__/`, `*.pyc`, `.venv/`, `.pytest_cache/`
  - `requirements.txt`: `fastapi`, `uvicorn[standard]`, `torch`, `transformers`, `pydantic`, `pytest`, `httpx`

- [ ] **Step 2**: Tulis `app/config.py`
```python
import os

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
DEVICE = os.environ.get("DEVICE", "auto")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
```

- [ ] **Step 3**: Verify import
```bash
cd 09-indobert-api && ~/.pyenv/versions/3.11.0/envs/dl-portfolio/bin/python -c "from app.config import ID2LABEL; print(ID2LABEL)"
```
Expected: `{0: 'negative', 1: 'neutral', 2: 'positive'}`

- [ ] **Step 4**: Commit
```bash
git add 09-indobert-api && git commit -m "feat: scaffold 09-indobert-api project config"
```

## Task 2: Schemas Pydantic

**Files:**
- Create: `app/schemas.py`

**Interfaces:**
- Produces: `PredictRequest{text}`, `PredictBatchRequest{texts}`, `PredictResponse{text,label,score,probabilities}`, `PredictBatchResponse{results}`, `HealthResponse`, `MetadataResponse`.

- [ ] **Step 1**: Tulis `app/schemas.py`
```python
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Teks yang akan diklasifikasi")


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="Daftar teks")


class PredictResponse(BaseModel):
    text: str
    label: str
    score: float
    probabilities: dict[str, float]


class PredictBatchResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    model_name: str
    model_type: str
    labels: list[str]
    max_length: int
```

- [ ] **Step 2**: Sanity import (sama seperti Task 1 Step 3, import `app.schemas`).
- [ ] **Step 3**: Commit
```bash
git add 09-indobert-api/app && git commit -m "feat: add pydantic schemas for API"
```

## Task 3: Inference wrapper + unit test

**Files:**
- Create: `app/inference.py`, `tests/test_inference.py`, `tests/__init__.py`

**Interfaces:**
- Consumes: `app.config` (DEVICE, MODEL_DIR, MAX_LENGTH, ID2LABEL), `AutoTokenizer`, `AutoModelForSequenceClassification`.
- Produces: `class SentimentModel` dengan `load()`, `predict(text)->dict`, `predict_batch(texts)->list[dict]`, property `loaded`, `device`. Helper `get_device()`, `softmax_scores(logits)`.

- [ ] **Step 1: Tulis test yang gagal** (`tests/test_inference.py`)
```python
import torch

from app.inference import SentimentModel, softmax_scores


def test_softmax_sums_to_one():
    logits = torch.tensor([1.0, 2.0, 3.0])
    scores = softmax_scores(logits)
    assert abs(float(scores.sum()) - 1.0) < 1e-6


class FakeTorchModel:
    def __call__(self, **tokens):
        return torch.tensor([[1.0, 2.0, 3.0]])  # logits


def _make_model():
    m = SentimentModel(model_dir="model")
    m.device = torch.device("cpu")
    m.tokenizer = _FakeTokenizer()
    m.model = FakeTorchModel()
    return m


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": torch.zeros(len(texts), 5, dtype=torch.long)}


def test_predict_maps_labels_and_probs():
    m = _make_model()
    result = m.predict("bagus sekali")
    assert result["label"] == "positive"
    assert 0 <= result["score"] <= 1
    assert set(result["probabilities"]) == {"negative", "neutral", "positive"}


def test_predict_batch_returns_same_length():
    m = _make_model()
    results = m.predict_batch(["jelek", "bagus"])
    assert len(results) == 2
    assert all(r["label"] in ("negative", "neutral", "positive") for r in results)
```

- [ ] **Step 2**: Run → FAIL (`app.inference` tidak ada).
  ```bash
  cd 09-indobert-api && ~/.pyenv/versions/3.11.0/envs/dl-portfolio/bin/python -m pytest tests/test_inference.py -v
  ```

- [ ] **Step 3**: Tulis `app/inference.py`
```python
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import ID2LABEL, LABELS, MAX_LENGTH, MODEL_DIR


def get_device():
    if os.environ.get("DEVICE") and os.environ["DEVICE"] != "auto":
        return torch.device(os.environ["DEVICE"])
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def softmax_scores(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1).squeeze(0)


class SentimentModel:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.device = get_device()
        self.tokenizer = None
        self.model = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id={v: k for k, v in ID2LABEL.items()},
        ).to(self.device)
        self.model.eval()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _tokenize(self, texts: list[str]):
        return self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        ).to(self.device)

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        tokens = self._tokenize(texts)
        logits = self.model(**tokens)
        logits = logits.logits if hasattr(logits, "logits") else logits
        probs = softmax_scores(logits)
        results = []
        for i, text in enumerate(texts):
            scores = {label: float(probs[i, idx]) for label, idx in ID2LABEL.items()}
            label = max(scores, key=scores.get)
            results.append({"text": text, "label": label, "score": scores[label], "probabilities": scores})
        return results
```

- [ ] **Step 4**: Run tests → PASS.
- [ ] **Step 5**: Commit
```bash
git add 09-indobert-api/app/inference.py 09-indobert-api/tests && git commit -m "feat: add sentiment inference wrapper"
```

## Task 4: FastAPI app + API tests

**Files:**
- Create: `app/main.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `SentimentModel`, schemas.
- Produces: app `create_app(model=None) -> FastAPI`; endpoint `/health`, `/metadata`, `/predict`, `/predict_batch`. Module-level `app = create_app()`.

- [ ] **Step 1: Tulis test yang gagal** (`tests/test_api.py`)
```python
import pytest
from fastapi.testclient import TestClient

from app.inference import SentimentModel
from app.main import create_app


def _make_client(model):
    app = create_app(model)
    with TestClient(app) as c:
        yield c


def test_health_ok():
    model = SentimentModel(model_dir="model")
    model.model = object()  # fake loaded
    model.device = torch.device("cpu")
    client = _make_client(model)
    r = next(client).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_503_when_not_loaded():
    model = SentimentModel(model_dir="model")
    model.model = None
    client = _make_client(model)
    r = next(client).get("/health")
    assert r.status_code == 503


def test_metadata_ok():
    model = SentimentModel(model_dir="model")
    model.model = object()
    client = _make_client(model)
    r = next(client).get("/metadata")
    assert r.status_code == 200
    assert r.json()["labels"] == ["negative", "neutral", "positive"]


def test_predict_returns_label(monkeypatch):
    model = SentimentModel(model_dir="model")
    model.model = object()

    def fake_predict(text):
        return {"text": text, "label": "positive", "score": 0.9,
                "probabilities": {"negative": 0.05, "neutral": 0.05, "positive": 0.9}}

    monkeypatch.setattr(model, "predict", fake_predict)
    client = _make_client(model)
    r = next(client).post("/predict", json={"text": "bagus sekali"})
    assert r.status_code == 200
    assert r.json()["label"] == "positive"


def test_predict_empty_text_422():
    model = SentimentModel(model_dir="model")
    model.model = object()
    client = _make_client(model)
    r = next(client).post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_batch_ok(monkeypatch):
    model = SentimentModel(model_dir="model")
    model.model = object()

    def fake_batch(texts):
        return [{"text": t, "label": "negative", "score": 0.8,
                 "probabilities": {"negative": 0.8, "neutral": 0.1, "positive": 0.1}} for t in texts]

    monkeypatch.setattr(model, "predict_batch", fake_batch)
    client = _make_client(model)
    r = next(client).post("/predict_batch", json={"texts": ["jelek", "buruk"]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2
```

Catatan: TestClient + lifespan memanggil `model.load()` bila `model.model is None` — karena stub model sudah set, load tidak terpanggil (guard di lifespan). Untuk `test_health_503_when_not_loaded`, `model.model = None` memicu load() → akan gagal (model dir tidak ada). **Ruling yang berlaku:** lifespan harus dihindari untuk test 503 — test memakai `create_app` dengan lifespan yang tidak me-load saat model.model sudah diset, ATAU test 503 di-handle dengan `model.loaded` check di endpoint (yang dipakai) — lihat implementasi main.py di bawah: `/health` cek `model.loaded` dan return 503, tanpa perlu load. Perlu penyesuaian: TestClient lifespan memanggil load() saat model.model is None. Solusi: buat lifespan hanya load jika `not model.loaded`; untuk test 503, gunakan `TestClient(app)` TANPA context manager (tanpa lifespan) sehingga tidak memicu load. Sesuaikan test sesuai itu.

- [ ] **Step 2**: Run → FAIL (`app.main` tidak ada).

- [ ] **Step 3**: Tulis `app/main.py`
```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.inference import SentimentModel
from app.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)


def create_app(model: SentimentModel | None = None) -> FastAPI:
    model = model or SentimentModel()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not model.loaded:
            model.load()
        yield

    app = FastAPI(title="IndoBERT Sentiment API", version="1.0.0", lifespan=lifespan)
    app.model = model

    @app.get("/health", response_model=HealthResponse)
    def health():
        if not model.loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {"status": "ok", "device": str(model.device), "model_loaded": True}

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata():
        return {
            "model_name": "indobert-large-p1",
            "model_type": "BertForSequenceClassification",
            "labels": ["negative", "neutral", "positive"],
            "max_length": 128,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        return model.predict(req.text)

    @app.post("/predict_batch", response_model=PredictBatchResponse)
    def predict_batch(req: PredictBatchRequest):
        return PredictBatchResponse(results=model.predict_batch(req.texts))

    return app


app = create_app()
```

- [ ] **Step 4**: Run tests → PASS. Sesuaikan test 503: gunakan TestClient tanpa context manager agar lifespan tidak memicu `model.load()`.
- [ ] **Step 5**: Commit
```bash
git add 09-indobert-api/app/main.py 09-indobert-api/tests && git commit -m "feat: add fastapi app with health predict endpoints"
```

## Task 5: Dockerfile + docker-compose

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `docker-compose.yml`

- [ ] **Step 1**: Tulis `Dockerfile`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2**: Tulis `.dockerignore`
```
__pycache__/
*.pyc
.pytest_cache/
tests/
bruno/
.git/
```

- [ ] **Step 3**: Tulis `docker-compose.yml`
```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ../08-indobert-finetuning/outputs/best_model:/app/model:ro
    environment:
      - DEVICE=cpu
    restart: unless-stopped
```

- [ ] **Step 4**: Build & run
```bash
cd 09-indobert-api && docker compose up --build -d && sleep 5 && curl -s http://localhost:8000/health
```
Expected: JSON `{"status":"ok","device":"cpu","model_loaded":true}`
(Catatan: build pertama lama — unduh torch+transformers di image. Bila build gagal karena network, catat di report dan lanjut; verifikasi Dockerfile minimal dengan `docker build` sukses.)

- [ ] **Step 5**: Stop & cleanup
```bash
docker compose down
```

- [ ] **Step 6**: Commit
```bash
git add 09-indobert-api/Dockerfile 09-indobert-api/.dockerignore 09-indobert-api/docker-compose.yml && git commit -m "feat: add dockerfile and compose for deployment"
```

## Task 6: Koleksi Bruno + README

**Files:**
- Create: `bruno/collection.json`, `README.md`

- [ ] **Step 1**: Tulis `bruno/collection.json` dengan 4 request: GET `/health`, GET `/metadata`, POST `/predict` (contoh `{"text":"filmnya bagus sekali"}`), POST `/predict_batch` (contoh 2 teks).
  Format Bruno v4 collection:
```json
{
  "version": "4",
  "name": "IndoBERT Sentiment API",
  "items": [
    {
      "type": "request",
      "name": "health",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/health"
      }
    },
    {
      "type": "request",
      "name": "metadata",
      "request": {
        "method": "GET",
        "url": "http://localhost:8000/metadata"
      }
    },
    {
      "type": "request",
      "name": "predict",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/predict",
        "headers": [{"name": "Content-Type", "value": "application/json"}],
        "body": {
          "mode": "json",
          "json": "{\"text\": \"filmnya bagus sekali\"}"
        }
      }
    },
    {
      "type": "request",
      "name": "predict_batch",
      "request": {
        "method": "POST",
        "url": "http://localhost:8000/predict_batch",
        "headers": [{"name": "Content-Type", "value": "application/json"}],
        "body": {
          "mode": "json",
          "json": "{\"texts\": [\"filmnya bagus sekali\", \"pelayanannya sangat buruk\"]}"
        }
      }
    }
  ]
}
```

- [ ] **Step 2**: Tulis `README.md` — penjelasan project, cara run lokal (`uvicorn app.main:app --reload`), cara run Docker (`docker compose up --build`), cara import koleksi ke Bruno, daftar endpoint dengan contoh request/response. Bahasa Indonesia.
- [ ] **Step 3**: Commit
```bash
git add 09-indobert-api/bruno 09-indobert-api/README.md && git commit -m "docs: add bruno collection and readme"
```

---

## Self-Review (dilakukan setelah menulis plan)

1. **Spec coverage:** Semua endpoint (health/predict/batch/metadata) di Task 4. Docker+compose di Task 5. Testing unit+API di Task 3-4. Bruno+README di Task 6. ✅
2. **Placeholder scan:** Tidak ada TBD/TODO. Semua kode konkret. ✅
3. **Type consistency:** `create_app(model=None)`, `SentimentModel.predict/predict_batch`, `softmax_scores` konsisten antar task. ✅
4. **Catatan test /health:** Test `test_health_503_when_not_loaded` perlu TestClient tanpa lifespan (context manager) agar tidak memicu `model.load()` pada model dir yang tidak ada. Ditandai di Task 4 Step 1 & 4.