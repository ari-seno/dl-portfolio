"""
Vocabulary builder for the SmSA dataset.

Builds a word -> index mapping from the training split only (never from
valid/test, to avoid leakage). Includes two special tokens:
  <pad> (index 0) - padding for batching variable-length sequences
  <unk> (index 1) - out-of-vocabulary words at train/inference time
"""

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List

from src.tokenizer import tokenize

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Vocab:
    def __init__(self, min_freq: int = 2):
        """
        Args:
            min_freq: minimum token frequency in training data to be
                included in the vocabulary. Tokens below this are
                mapped to <unk>. Helps control vocab size / noise from
                typos and rare words.
        """
        self.min_freq = min_freq
        self.token2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2token = {0: PAD_TOKEN, 1: UNK_TOKEN}

    def build(self, texts: Iterable[str]) -> "Vocab":
        """Build vocabulary from an iterable of raw (untokenized) texts."""
        counter = Counter()
        for text in texts:
            counter.update(tokenize(text))

        for token, freq in counter.most_common():
            if freq < self.min_freq:
                continue
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

        return self

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert a list of tokens to a list of indices (<unk> if OOV)."""
        unk_idx = self.token2idx[UNK_TOKEN]
        return [self.token2idx.get(tok, unk_idx) for tok in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert a list of indices back to tokens."""
        return [self.idx2token.get(idx, UNK_TOKEN) for idx in indices]

    def __len__(self) -> int:
        return len(self.token2idx)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"min_freq": self.min_freq, "token2idx": self.token2idx},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data["min_freq"])
        vocab.token2idx = data["token2idx"]
        vocab.idx2token = {int(idx): tok for tok, idx in vocab.token2idx.items()}
        return vocab


if __name__ == "__main__":
    import pandas as pd

    train_path = Path(__file__).resolve().parent.parent / "data" / "train_preprocess.tsv"
    df = pd.read_csv(train_path, sep="\t", header=None, names=["text", "label"])

    vocab = Vocab(min_freq=2).build(df["text"])
    print(f"Vocab size (min_freq=2): {len(vocab)}")

    out_path = Path(__file__).resolve().parent.parent / "outputs" / "vocab.json"
    vocab.save(str(out_path))
    print(f"Saved to {out_path}")