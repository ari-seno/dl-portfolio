# src/model.py

from transformers import AutoModelForSequenceClassification

from src.dataset import DEFAULT_MODEL_NAME, ID2LABEL, LABEL2ID


def build_model(model_name: str = DEFAULT_MODEL_NAME):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


if __name__ == "__main__":
    model = build_model()
    print(model.config)