import torch
from fastapi.testclient import TestClient

from app.inference import SentimentModel
from app.main import create_app


def _make_client(model):
    app = create_app(model)
    return TestClient(app)


def test_health_ok():
    model = SentimentModel(model_dir="model")
    model.model = object()  # fake loaded
    model.device = torch.device("cpu")
    r = _make_client(model).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_503_when_not_loaded():
    model = SentimentModel(model_dir="model")
    model.model = None
    r = _make_client(model).get("/health")
    assert r.status_code == 503


def test_metadata_ok():
    model = SentimentModel(model_dir="model")
    model.model = object()
    r = _make_client(model).get("/metadata")
    assert r.status_code == 200
    assert r.json()["labels"] == ["negative", "neutral", "positive"]


def test_predict_returns_label(monkeypatch):
    model = SentimentModel(model_dir="model")
    model.model = object()

    def fake_predict(text):
        return {"text": text, "label": "positive", "score": 0.9,
                "probabilities": {"negative": 0.05, "neutral": 0.05, "positive": 0.9}}

    monkeypatch.setattr(model, "predict", fake_predict)
    r = _make_client(model).post("/predict", json={"text": "bagus sekali"})
    assert r.status_code == 200
    assert r.json()["label"] == "positive"


def test_predict_empty_text_422():
    model = SentimentModel(model_dir="model")
    model.model = object()
    r = _make_client(model).post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_503_when_not_loaded():
    model = SentimentModel(model_dir="model")
    model.model = None
    r = _make_client(model).post("/predict", json={"text": "bagus"})
    assert r.status_code == 503


def test_predict_batch_503_when_not_loaded():
    model = SentimentModel(model_dir="model")
    model.model = None
    r = _make_client(model).post("/predict_batch", json={"texts": ["bagus"]})
    assert r.status_code == 503


def test_predict_batch_ok(monkeypatch):
    model = SentimentModel(model_dir="model")
    model.model = object()

    def fake_batch(texts):
        return [{"text": t, "label": "negative", "score": 0.8,
                 "probabilities": {"negative": 0.8, "neutral": 0.1, "positive": 0.1}} for t in texts]

    monkeypatch.setattr(model, "predict_batch", fake_batch)
    r = _make_client(model).post("/predict_batch", json={"texts": ["jelek", "buruk"]})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_predict_batch_empty_element_422():
    model = SentimentModel(model_dir="model")
    model.model = object()
    r = _make_client(model).post("/predict_batch", json={"texts": [""]})
    assert r.status_code == 422


def test_predict_batch_blank_element_422():
    model = SentimentModel(model_dir="model")
    model.model = object()
    r = _make_client(model).post("/predict_batch", json={"texts": ["   "]})
    assert r.status_code == 422


def test_predict_batch_too_many_422():
    model = SentimentModel(model_dir="model")
    model.model = object()
    r = _make_client(model).post("/predict_batch", json={"texts": ["x"] * 101})
    assert r.status_code == 422
