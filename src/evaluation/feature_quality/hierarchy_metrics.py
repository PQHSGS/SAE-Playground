from typing import Dict
import torch
from src.core.base_dictionary import BaseDictionary


@torch.no_grad()
def compute_hierarchy_and_monosemanticity(
    dictionary_model: BaseDictionary,
    activations: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes Feature Monosemanticity and Hierarchical Sparseness Statistics:
    - Gini Sparseness Coefficient: Measures the inequality of feature activations across tokens.
      A high Gini index (approaching 1.0) signifies extreme selectivity (monosemanticity).
    - Excess Kurtosis: High kurtosis indicates heavy-tailed, highly selective activation patterns.
    - Mean Activation Skewness: Positive skew indicates sharp peak activations on rare tokens.
    """
    device = next(dictionary_model.parameters()).device
    activations = activations.to(device)

    f = dictionary_model.encode(activations)  # (N, d_sae)
    if f.ndim > 2:
        f = f.view(-1, f.shape[-1])
    num_tokens, d_sae = f.shape

    # 1. Gini coefficient of feature activation distribution across all tokens
    # Sample subset of active features for fast computation
    active_feats = f[:, (f > 0).any(dim=0)]
    if active_feats.shape[1] > 0:
        sorted_acts, _ = torch.sort(active_feats, dim=0)
        n = num_tokens
        index = torch.arange(1, n + 1, device=device, dtype=torch.float32).unsqueeze(1)
        sum_acts = sorted_acts.sum(dim=0, keepdim=True) + 1e-8
        gini_per_feature = ((2.0 * index - n - 1) * sorted_acts).sum(dim=0) / (n * sum_acts)
        mean_gini = gini_per_feature.mean().item()
    else:
        mean_gini = 0.0

    # 2. Kurtosis and Skewness of activations
    mean = f.mean(dim=0, keepdim=True)
    std = f.std(dim=0, keepdim=True) + 1e-8
    norm_f = (f - mean) / std

    skewness = (norm_f ** 3).mean(dim=0).mean().item()
    excess_kurtosis = ((norm_f ** 4).mean(dim=0) - 3.0).mean().item()

    return {
        "mean_gini_sparseness": mean_gini,
        "mean_activation_skewness": skewness,
        "mean_excess_kurtosis": excess_kurtosis,
    }
