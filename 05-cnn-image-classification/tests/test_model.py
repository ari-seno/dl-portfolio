"""
Unit tests for CIFAR10CNN (src/model.py).

Covers: output shape, no NaN/Inf in forward pass, trainable parameter
count, behavior across different batch sizes, and eval-mode determinism
(BatchNorm/Dropout disabled -> same input gives same output).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import CIFAR10CNN


@pytest.fixture
def model():
    torch.manual_seed(42)
    return CIFAR10CNN(num_classes=10, dropout=0.5)


def test_output_shape(model):
    x = torch.randn(8, 3, 32, 32)
    out = model(x)
    assert out.shape == (8, 10)


def test_output_no_nan_or_inf(model):
    x = torch.randn(8, 3, 32, 32)
    out = model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_trainable_parameters_exist(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0


def test_different_batch_sizes(model):
    for batch_size in [1, 4, 16, 32]:
        x = torch.randn(batch_size, 3, 32, 32)
        out = model(x)
        assert out.shape == (batch_size, 10)


def test_eval_mode_determinism(model):
    model.eval()
    x = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2)


def test_train_mode_uses_dropout(model):
    # In train mode, Dropout is stochastic, so two forward passes on the
    # same input should generally differ (sanity check dropout is active).
    model.train()
    x = torch.randn(4, 3, 32, 32)

    torch.manual_seed(0)
    out1 = model(x)
    torch.manual_seed(1)
    out2 = model(x)

    assert not torch.allclose(out1, out2)


def test_gradients_flow(model):
    x = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    criterion = torch.nn.CrossEntropyLoss()

    out = model(x)
    loss = criterion(out, labels)
    loss.backward()

    grad_norms = [
        p.grad.norm().item() for p in model.parameters() if p.grad is not None
    ]
    assert len(grad_norms) > 0
    assert all(g >= 0 for g in grad_norms)
    assert any(g > 0 for g in grad_norms)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))