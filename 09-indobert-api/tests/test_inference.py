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
