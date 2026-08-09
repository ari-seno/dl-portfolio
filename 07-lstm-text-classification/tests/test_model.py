"""
Unit tests for LSTMClassifier (src/model.py).

Mirrors the testing pattern used across the portfolio (Projects 3-6):
output shape, no NaN/Inf, trainable params exist, different batch sizes,
eval-mode determinism, gradients flow — plus tests specific to this
project's variable-length sequence handling (pack_padded_sequence).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model import LSTMClassifier


VOCAB_SIZE = 500
NUM_CLASSES = 3


def make_batch(batch_size=4, min_len=3, max_len=15, vocab_size=VOCAB_SIZE):
    """Builds a random padded batch + lengths, mimicking collate_fn output."""
    lengths = torch.randint(min_len, max_len + 1, (batch_size,))
    max_seq_len = lengths.max().item()
    texts = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    for i, length in enumerate(lengths):
        texts[i, :length] = torch.randint(2, vocab_size, (length,))  # avoid pad(0)/unk(1)
    return texts, lengths


class TestLSTMClassifier:
    def test_output_shape(self):
        model = LSTMClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
        texts, lengths = make_batch(batch_size=8)
        logits = model(texts, lengths)
        assert logits.shape == (8, NUM_CLASSES)

    def test_no_nan_or_inf(self):
        model = LSTMClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
        texts, lengths = make_batch(batch_size=8)
        logits = model(texts, lengths)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_trainable_params_exist(self):
        model = LSTMClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_different_batch_sizes(self):
        model = LSTMClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
        for batch_size in [1, 4, 16]:
            texts, lengths = make_batch(batch_size=batch_size)
            logits = model(texts, lengths)
            assert logits.shape == (batch_size, NUM_CLASSES)

    def test_variable_sequence_lengths(self):
        """Sanity check specific to this project: sequences of very
        different lengths in the same batch (relies on pack_padded_sequence
        to correctly ignore padding) should not error or produce NaNs."""
        model = LSTMClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
        lengths = torch.tensor([2, 20, 5, 50])
        max_len = lengths.max().item()
        texts = torch.zeros((4, max_len), dtype=torch.long)
        for i, length in enumerate(lengths):
            texts[i, :length] = torch.randint(2, VOCAB_SIZE, (length,))
        logits = model(texts, lengths)
        assert logits.shape == (4, NUM_CLASSES)
        assert not torch.isnan(logits).any()

    def test_eval_mode_determinism(self):
        """In eval mode (dropout off), two forward passes on the same
        input should produce identical output."""
        model = LSTMClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES, dropout=0.5)
        model.eval()
        texts, lengths = make_batch(batch_size=4)
        with torch.no_grad():
            out1 = model(texts, lengths)
            out2 = model(texts, lengths)
        assert torch.allclose(out1, out2)

    def test_train_mode_dropout_stochasticity(self):
        """In train mode with dropout > 0, two forward passes on the
        same input should (almost certainly) differ."""
        model = LSTMClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES, dropout=0.5)
        model.train()
        texts, lengths = make_batch(batch_size=4)
        out1 = model(texts, lengths)
        out2 = model(texts, lengths)
        assert not torch.allclose(out1, out2)

    def test_gradients_flow(self):
        model = LSTMClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
        texts, lengths = make_batch(batch_size=4)
        logits = model(texts, lengths)
        loss = logits.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_unidirectional_variant(self):
        """bidirectional=False should still produce correctly-shaped output
        (uses a different code path for extracting the final hidden state)."""
        model = LSTMClassifier(
            vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES, bidirectional=False
        )
        texts, lengths = make_batch(batch_size=4)
        logits = model(texts, lengths)
        assert logits.shape == (4, NUM_CLASSES)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])