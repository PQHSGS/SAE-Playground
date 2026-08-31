import os
import tempfile
import pytest
import torch
from src.architectures.registry import build_dictionary, DICTIONARY_REGISTRY


@pytest.mark.parametrize("arch", [
    "standard", "topk", "batch_topk", "jumprelu", "gated", "treesae", "sasa", "matryoshka", "matching_pursuit"
])
def test_sae_forward_backward(arch):
    d_in = 64
    d_sae = 256
    batch_size = 16

    kwargs = {"k": 8}
    if arch == "jumprelu":
        kwargs["init_threshold"] = 0.01

    model = build_dictionary(arch, d_in=d_in, d_sae=d_sae, **kwargs)
    x = torch.randn(batch_size, d_in, requires_grad=True)

    out = model(x)
    assert out.reconstructed.shape == (batch_size, d_in)
    assert out.feature_acts.shape == (batch_size, d_sae)
    assert out.loss is not None

    # Backward pass
    out.loss.backward()
    assert model.get_decoder_weights().grad is not None

    # Normalization check
    model.normalize_decoder_weights()
    w_dec = model.get_decoder_weights()
    norms = torch.norm(w_dec, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


@pytest.mark.parametrize("arch", ["transcoder", "skip_transcoder", "gated_transcoder"])
def test_transcoder_architectures(arch):
    d_in = 64
    d_out = 64
    d_sae = 256
    batch_size = 16

    model = build_dictionary(arch, d_in=d_in, d_sae=d_sae, d_out=d_out, k=8)
    x_in = torch.randn(batch_size, d_in)
    y_target = torch.randn(batch_size, d_out)

    out = model(x_in, target=y_target)
    assert out.reconstructed.shape == (batch_size, d_out)
    assert out.feature_acts.shape == (batch_size, d_sae)
    assert out.loss is not None

    out.loss.backward()
    assert model.get_decoder_weights().grad is not None


@pytest.mark.parametrize("arch", ["crosscoder", "batch_topk_crosscoder"])
def test_crosscoder_architectures(arch):
    n_layers = 3
    d_in = 64
    d_sae = 256
    batch_size = 16

    model = build_dictionary(arch, d_in=d_in, d_sae=d_sae, n_layers=n_layers, k=16)
    x = torch.randn(batch_size, n_layers, d_in)

    out = model(x)
    assert out.reconstructed.shape == (batch_size, n_layers, d_in)
    assert out.feature_acts.shape == (batch_size, d_sae)
    assert out.loss is not None

    out.loss.backward()
    assert model.get_decoder_weights().grad is not None


def test_save_and_load_roundtrip():
    d_in = 32
    d_sae = 128
    model = build_dictionary("topk", d_in=d_in, d_sae=d_sae, k=4)
    x = torch.randn(8, d_in)
    orig_out = model(x).reconstructed

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded = type(model).from_pretrained(tmpdir)
        loaded_out = loaded(x).reconstructed
        assert torch.allclose(orig_out, loaded_out, atol=1e-5)
