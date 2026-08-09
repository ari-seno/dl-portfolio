"""
PyTorch Dataset for the SmSA sentiment classification dataset.

Reads a train/valid/test TSV file (text \t label, no header), tokenizes
and encodes the text using a pre-built Vocab, and maps the 3 string
labels to integer class indices. Padding is done per-batch (not
per-dataset) via `collate_fn`, so short sentences aren't wastefully
padded to the length of the longest sentence in the whole dataset.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.tokenizer import tokenize
from src.vocab import Vocab, PAD_TOKEN

LABEL2IDX = {"negative": 0, "neutral": 1, "positive": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


class SmSADataset(Dataset):
    def __init__(self, tsv_path: str, vocab: Vocab):
        """
        Args:
            tsv_path: path to a SmSA TSV split (train/valid/test).
            vocab: a Vocab already built from the training set.
        """
        self.vocab = vocab
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["text", "label"])
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        tokens = tokenize(self.texts[idx])
        encoded = self.vocab.encode(tokens)
        # guard against empty-token edge case (e.g. a blank/whitespace-only row)
        if len(encoded) == 0:
            encoded = [self.vocab.token2idx.get(PAD_TOKEN, 0)]
        label_idx = LABEL2IDX[self.labels[idx]]
        return torch.tensor(encoded, dtype=torch.long), label_idx


def collate_fn(
    batch: List[Tuple[torch.Tensor, int]], pad_idx: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads a batch of variable-length sequences to the same length.

    Returns:
        padded_texts: (batch_size, max_seq_len)
        lengths: (batch_size,) original (unpadded) length of each sequence
                 - needed for nn.utils.rnn.pack_padded_sequence in model.py
        labels: (batch_size,)
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    root = Path(__file__).resolve().parent.parent
    vocab = Vocab.load(str(root / "outputs" / "vocab.json"))

    train_ds = SmSADataset(str(root / "data" / "train_preprocess.tsv"), vocab)
    print(f"Train dataset size: {len(train_ds)}")

    sample_text, sample_label = train_ds[0]
    print(f"Sample encoded text (len={len(sample_text)}): {sample_text[:10]}...")
    print(f"Sample label: {sample_label} ({IDX2LABEL[sample_label]})")

    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    padded, lengths, labels = next(iter(loader))
    print(f"\nBatch shapes -> padded: {padded.shape}, lengths: {lengths}, labels: {labels}")