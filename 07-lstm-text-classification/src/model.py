"""
BiLSTM text classifier for SmSA sentiment analysis.

Architecture:
    Embedding (trained from scratch, see roadmap decision notes)
    -> Bidirectional LSTM (packed sequence, ignores padding)
    -> concat final forward + backward hidden states
    -> Dropout -> Linear -> 3-class logits (negative/neutral/positive)
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.3,
        pad_idx: int = 0,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            # dropout between LSTM layers only applies when num_layers > 1
            dropout=dropout if num_layers > 1 else 0.0,
        )

        num_directions = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * num_directions, num_classes)

    def forward(self, texts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            texts: (batch_size, seq_len) padded token indices
            lengths: (batch_size,) true (unpadded) length of each sequence

        Returns:
            logits: (batch_size, num_classes)
        """
        embedded = self.embedding(texts)  # (batch, seq_len, embedding_dim)

        # pack so the LSTM skips over padding entirely instead of
        # treating <pad> tokens as real input
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # hidden: (num_layers * num_directions, batch, hidden_dim)

        if self.lstm.bidirectional:
            # last layer's forward and backward final hidden states
            hidden_fwd = hidden[-2, :, :]
            hidden_bwd = hidden[-1, :, :]
            final_hidden = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        else:
            final_hidden = hidden[-1, :, :]

        out = self.dropout(final_hidden)
        logits = self.fc(out)
        return logits


if __name__ == "__main__":
    from pathlib import Path

    from torch.utils.data import DataLoader

    from src.dataset import SmSADataset, collate_fn
    from src.vocab import Vocab

    root = Path(__file__).resolve().parent.parent
    vocab = Vocab.load(str(root / "outputs" / "vocab.json"))

    train_ds = SmSADataset(str(root / "data" / "train_preprocess.tsv"), vocab)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    padded, lengths, labels = next(iter(loader))

    model = LSTMClassifier(vocab_size=len(vocab))
    logits = model(padded, lengths)

    print(f"Input shape : {padded.shape}")
    print(f"Logits shape: {logits.shape}  (expected: [8, 3])")
    print(f"Sample logits[0]: {logits[0]}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")