import pytest
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.auto_interpret.sample_collector import FeatureSampleCollector
from src.auto_interpret.simulator import FeatureSimulator
from src.architectures.sae.topk_sae import TopKSAE
from src.circuits.attribution_patching import EdgeAttributionPatching
from src.utils.hf_helpers import compute_direct_logit_attribution


def test_direct_logit_attribution():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    d_model = 64
    vocab_size = 1000

    decoder_vector = torch.randn(d_model)
    unembedding = torch.randn(vocab_size, d_model)

    promoted, suppressed = compute_direct_logit_attribution(
        decoder_vector=decoder_vector,
        unembedding_weights=unembedding,
        tokenizer=tokenizer,
        top_k=5
    )

    assert len(promoted) == 5
    assert len(suppressed) == 5
    assert promoted[0]["logit"] >= promoted[1]["logit"]
    assert suppressed[0]["logit"] <= suppressed[1]["logit"]


def test_feature_simulator_correlation():
    actual = [0.0, 1.0, 2.0, 3.0, 4.0]
    predicted = [0.1, 1.1, 1.9, 3.2, 4.1]
    score = FeatureSimulator.compute_faithfulness_score(actual, predicted)
    assert score > 0.95


def test_edge_attribution_patching_gradient_flow():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    src_sae = TopKSAE(d_in=768, d_sae=128, k=8).to(device)
    tgt_sae = TopKSAE(d_in=768, d_sae=128, k=8).to(device)

    eap = EdgeAttributionPatching(
        model=model,
        tokenizer=tokenizer,
        source_sae=src_sae,
        target_sae=tgt_sae,
        source_hook="transformer.h.2",
        target_hook="transformer.h.4",
    )

    result = eap.compute_circuit_attributions(
        clean_text="The Eiffel Tower is in Paris",
        corrupted_text="The Colosseum is in Rome",
        target_token_id=tokenizer.encode(" Paris")[0],
        top_k_edges=10,
    )

    assert result.source_attributions.shape == (128,)
    assert result.target_attributions.shape == (128,)


def test_transcoder_circuit_replacement_and_graph():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    from src.architectures.transcoders.skip_transcoder import SkipTranscoder
    from src.circuits.transcoder_circuit import TranscoderCircuitGraph

    # Create SkipTranscoders for 2 layers
    t0 = SkipTranscoder(d_in=768, d_sae=64, d_out=768).to(device)
    t1 = SkipTranscoder(d_in=768, d_sae=64, d_out=768).to(device)

    hook_pairs = {
        "transformer.h.0.mlp": "transformer.h.0.mlp.c_proj",
        "transformer.h.1.mlp": "transformer.h.1.mlp.c_proj",
    }

    graph = TranscoderCircuitGraph(
        model=model,
        tokenizer=tokenizer,
        transcoders={
            "transformer.h.0.mlp": t0,
            "transformer.h.1.mlp": t1,
        },
        hook_pairs=hook_pairs,
    )

    # 1. Test sublayer replacement faithfulness
    faith_res = graph.evaluate_replacement_faithfulness(["The Eiffel Tower is in Paris, France."])
    assert "faithfulness" in faith_res
    assert "baseline_loss" in faith_res
    assert "replaced_loss" in faith_res

    # 2. Test cross-layer attribution graph computation
    result = graph.compute_transcoder_attributions(
        clean_text="The Eiffel Tower is in Paris",
        corrupted_text="The Colosseum is in Rome",
        target_token_id=tokenizer.encode(" Paris")[0],
        top_k_edges=5,
    )

    assert result.baseline_loss > 0
    assert len(result.layer_feature_attributions) == 2
    assert len(result.top_edges) <= 5

    # 3. Test graph export
    exported = graph.export_graph_json(result)
    assert "nodes" in exported
    assert "edges" in exported

