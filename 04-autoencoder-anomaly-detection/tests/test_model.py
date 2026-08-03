import pytest
import torch

from src.model import Autoencoder


@pytest.fixture
def model():
    return Autoencoder(input_dim=30, latent_dim=6)


def test_output_shape_matches_input_shape(model):
    x = torch.randn(16, 30)
    output = model(x)
    assert output.shape == x.shape


def test_encoder_output_is_latent_dim(model):
    x = torch.randn(16, 30)
    latent = model.encoder(x)
    assert latent.shape == (16, 6)


def test_decoder_reconstructs_input_dim(model):
    latent = torch.randn(16, 6)
    reconstructed = model.decoder(latent)
    assert reconstructed.shape == (16, 30)


def test_forward_pass_no_nan_or_inf(model):
    x = torch.randn(8, 30)
    output = model(x)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_model_has_trainable_parameters(model):
    params = list(model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_different_latent_dims():
    model_small = Autoencoder(input_dim=30, latent_dim=4)
    model_large = Autoencoder(input_dim=30, latent_dim=10)

    x = torch.randn(8, 30)
    out_small = model_small(x)
    out_large = model_large(x)

    assert out_small.shape == (8, 30)
    assert out_large.shape == (8, 30)


def test_eval_mode_deterministic(model):
    model.eval()
    x = torch.randn(4, 30)

    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)

    assert torch.allclose(output1, output2)