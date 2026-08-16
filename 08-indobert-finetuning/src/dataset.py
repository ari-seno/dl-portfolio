# src/dataset.py

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

DEFAULT_MODEL_NAME = "indobenchmark/indobert-base-p1"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MAX_LENGTH = 128  # cukup untuk kalimat pendek SmSA (rata-rata < 50 token)


def load_raw_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    df["label"] = df["label"].map(LABEL2ID)
    return df


def build_dataset_dict(data_dir: str = "data") -> DatasetDict:
    splits = {
        "train": load_raw_df(f"{data_dir}/train_preprocess.tsv"),
        "validation": load_raw_df(f"{data_dir}/valid_preprocess.tsv"),
        "test": load_raw_df(f"{data_dir}/test_preprocess.tsv"),
    }
    return DatasetDict({
        name: Dataset.from_pandas(df, preserve_index=False)
        for name, df in splits.items()
    })


def compute_class_weights(dataset: Dataset, labels: list[str] | None = None) -> list[float]:
    """Balanced class weights: n_samples / (n_classes * n_in_class)."""
    labels = labels or sorted(LABEL2ID.values())
    counts = np.array([sum(1 for lbl in dataset["label"] if lbl == c) for c in labels])
    n_samples = counts.sum()
    return (n_samples / (len(labels) * counts)).astype(float).tolist()


def get_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(dataset_dict: DatasetDict, tokenizer, max_length: int = MAX_LENGTH) -> DatasetDict:
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset_dict.map(_tokenize, batched=True)


if __name__ == "__main__":
    raw = build_dataset_dict()
    tokenizer = get_tokenizer()
    tokenized = tokenize_dataset(raw, tokenizer)
    print(tokenized)
    print("Class weights:", compute_class_weights(tokenized["train"]))
    print(tokenized["train"][0])