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
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    def predict(self, text: str) -> dict:
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict]:
        tokens = self._tokenize(texts)
        logits = self.model(**tokens)
        logits = logits.logits if hasattr(logits, "logits") else logits
        probs = softmax_scores(logits)
        if probs.ndim == 1:
            probs = probs.unsqueeze(0)
        probs = probs.expand(len(texts), -1)
        results = []
        for i, text in enumerate(texts):
            scores = {label: float(probs[i, idx]) for idx, label in ID2LABEL.items()}
            label = max(scores, key=scores.get)
            results.append({"text": text, "label": label, "score": scores[label], "probabilities": scores})
        return results
