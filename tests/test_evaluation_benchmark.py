import torch
from src.architectures.sae.batch_topk_sae import BatchTopKSAE
from src.evaluation.benchmark_suite import SAEBenchmarkSuite


def test_sae_benchmark_suite():
    d_in = 64
    d_sae = 256
    sae = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=8)
    
    activations = torch.randn(100, d_in)
    bench = SAEBenchmarkSuite()
    
    results = bench.evaluate(
        dictionary_model=sae,
        activations=activations,
    )
    
    assert "reconstruction" in results
    assert "density_spectrum" in results
    assert "feature_splitting" in results
    assert "monosemanticity" in results
    
    assert 0.0 <= results["reconstruction"]["fraction_variance_explained"] <= 1.0
    assert 0.0 <= results["feature_splitting"]["mutual_coherence_max"] <= 1.0
    assert results["density_spectrum"]["total_features"] == d_sae
    assert "mean_gini_sparseness" in results["monosemanticity"]
