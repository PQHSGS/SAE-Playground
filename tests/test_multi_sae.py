import os
import tempfile
import torch
import pytest

from src.architectures.registry import build_dictionary
from src.core.multi_dictionary import MultiLayerDictionary
from src.core.config import TrainingConfig
from src.training.trainer import DictionaryTrainer


def test_multi_layer_dictionary_forward_and_save():
    d_in = 64
    d_sae = 256
    batch_size = 8

    # Create 3 layer dictionaries
    dict_map = {
        "model.layers.0": build_dictionary("topk", d_in=d_in, d_sae=d_sae, k=8),
        "model.layers.4": build_dictionary("batch_topk", d_in=d_in, d_sae=d_sae, k=8),
        "model.layers.8": build_dictionary("topk", d_in=d_in, d_sae=d_sae, k=8),
    }

    multi_dict = MultiLayerDictionary(dict_map)
    assert len(multi_dict.dictionaries) == 3

    # Generate synthetic multi-layer activations
    acts_dict = {
        "model.layers.0": torch.randn(batch_size, d_in),
        "model.layers.4": torch.randn(batch_size, d_in),
        "model.layers.8": torch.randn(batch_size, d_in),
    }

    # Forward pass
    total_loss, outputs = multi_dict(acts_dict)
    assert total_loss.item() > 0.0
    assert len(outputs) == 3
    assert outputs["model.layers.0"].reconstructed.shape == (batch_size, d_in)
    assert outputs["model.layers.4"].feature_acts.shape == (batch_size, d_sae)

    # Encode & Decode
    encoded = multi_dict.encode(acts_dict)
    decoded = multi_dict.decode(encoded)
    assert decoded["model.layers.8"].shape == (batch_size, d_in)

    # Test polymorphic DictionaryTrainer step
    cfg = TrainingConfig(
        dataset_path="dummy",
        batch_size=batch_size,
        total_steps=10,
        checkpoint_steps=5,
        output_dir="dummy_out",
    )
    trainer = DictionaryTrainer(multi_dict, activation_buffer=None, config=cfg, device="cpu")
    metrics = trainer.train_step(acts_dict)
    assert "loss/total" in metrics
    assert "layers/model_layers_0/nmse" in metrics
    assert "metrics/mean_l0" in metrics

    # Normalize decoder weights
    multi_dict.normalize_decoder_weights()

    # Save and Load Roundtrip
    with tempfile.TemporaryDirectory() as tmpdir:
        multi_dict.save_pretrained(tmpdir)
        loaded = MultiLayerDictionary.from_pretrained(tmpdir)

        # Check each layer dictionary loaded correctly
        assert len(loaded.dictionaries) == 3
        layer0_loaded = loaded.get_dictionary("model.layers.0")
        assert layer0_loaded.d_sae == d_sae

        # Test individual layer loadability
        from src.architectures.sae.topk_sae import TopKSAE
        layer0_direct = TopKSAE.from_pretrained(os.path.join(tmpdir, "model_layers_0"))
        assert layer0_direct.d_in == d_in


def test_multi_layer_transcoder_forward_and_step():
    d_in = 64
    d_sae = 256
    batch_size = 8

    # Create 2 layer transcoders (e.g. MLP 0 and MLP 1)
    dict_map = {
        "model.layers.0.mlp.in": build_dictionary("skip_transcoder", d_in=d_in, d_sae=d_sae, d_out=d_in),
        "model.layers.1.mlp.in": build_dictionary("transcoder", d_in=d_in, d_sae=d_sae, d_out=d_in),
    }
    multi_transcoder = MultiLayerDictionary(dict_map)
    assert len(multi_transcoder) == 2

    # Paired inputs and targets for each layer
    transcoder_batch = {
        "model.layers.0.mlp.in": (torch.randn(batch_size, d_in), torch.randn(batch_size, d_in)),
        "model.layers.1.mlp.in": (torch.randn(batch_size, d_in), torch.randn(batch_size, d_in)),
    }

    cfg = TrainingConfig(batch_size=batch_size, total_steps=5, output_dir="dummy_out")
    trainer = DictionaryTrainer(multi_transcoder, activation_buffer=None, config=cfg, device="cpu")
    metrics = trainer.train_step(transcoder_batch)

    assert "loss/total" in metrics
    assert "layers/model_layers_0_mlp_in/nmse" in metrics
    assert "layers/model_layers_1_mlp_in/nmse" in metrics
    assert metrics["loss/total"] > 0.0
