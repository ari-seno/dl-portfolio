import torch
import pytest

from src.model import FraudMLP


def test_output_shape():
    model = FraudMLP(input_dim=29)
    X = torch.randn(16, 29)
    out = model(X)
    assert out.shape == (16, 1)


def test_output_range():
    """Sigmoid output must be in [0, 1]."""
    model = FraudMLP(input_dim=29)
    model.eval()
    X = torch.randn(32, 29)
    with torch.no_grad():
        out = model(X)
    assert torch.all(out >= 0) and torch.all(out <= 1)


def test_batchnorm_requires_batch_gt_1_in_train_mode():
    """BatchNorm1d fails on batch size 1 in train mode (expected PyTorch behavior)."""
    model = FraudMLP(input_dim=29)
    model.train()
    X = torch.randn(1, 29)
    with pytest.raises(ValueError):
        model(X)


def test_eval_mode_works_with_batch_size_1():
    """In eval mode, BatchNorm uses running stats, so batch size 1 is fine."""
    model = FraudMLP(input_dim=29)
    model.eval()
    X = torch.randn(1, 29)
    with torch.no_grad():
        out = model(X)
    assert out.shape == (1, 1)


def test_gradients_flow():
    """Backward pass should populate gradients for all parameters."""
    model = FraudMLP(input_dim=29)
    model.train()
    X = torch.randn(8, 29)
    y = torch.randint(0, 2, (8, 1)).float()

    out = model(X)
    loss = torch.nn.functional.binary_cross_entropy(out, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


def test_different_input_dim():
    """Model should work with a different input_dim if configured."""
    model = FraudMLP(input_dim=10)
    X = torch.randn(4, 10)
    out = model(X)
    assert out.shape == (4, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])