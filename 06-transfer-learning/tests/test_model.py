import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest

from model import CIFAR10ResNet18


@pytest.fixture
def model():
    return CIFAR10ResNet18(num_classes=10, freeze_backbone=True)


def test_output_shape(model):
    x = torch.randn(4, 3, 128, 128)
    out = model(x)
    assert out.shape == (4, 10)


def test_no_nan_or_inf(model):
    x = torch.randn(4, 3, 128, 128)
    out = model(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_trainable_params_exist(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable) > 0


def test_backbone_frozen_by_default(model):
    # With freeze_backbone=True, backbone params should NOT require grad,
    # but the classifier head should.
    backbone_trainable = any(
        p.requires_grad for name, p in model.named_parameters()
        if not name.startswith("backbone.fc")
    )
    head_trainable = all(
        p.requires_grad for name, p in model.named_parameters()
        if name.startswith("backbone.fc")
    )
    assert backbone_trainable is False
    assert head_trainable is True


def test_unfreeze_backbone(model):
    model.unfreeze_backbone()
    all_trainable = all(p.requires_grad for p in model.parameters())
    assert all_trainable is True


def test_different_batch_sizes(model):
    for batch_size in [1, 8, 32]:
        x = torch.randn(batch_size, 3, 128, 128)
        out = model(x)
        assert out.shape == (batch_size, 10)


def test_eval_mode_determinism(model):
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    out1 = model(x)
    out2 = model(x)
    assert torch.allclose(out1, out2)


def test_gradients_flow(model):
    x = torch.randn(4, 3, 128, 128)
    target = torch.randint(0, 10, (4,))
    criterion = torch.nn.CrossEntropyLoss()

    out = model(x)
    loss = criterion(out, target)
    loss.backward()

    # Only params that require grad should have gradients
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None