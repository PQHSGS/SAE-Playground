from typing import Dict
import torch
from src.core.base_dictionary import BaseDictionary


@torch.no_grad()
def compute_feature_statistics(
    dictionary_model: BaseDictionary,
    activations: torch.Tensor,
    dead_threshold_frac: float = 1e-6,
) -> Dict[str, any]:
    """
    Computes global feature firing distributions:
    - Dead feature count & percentage
    - Highly active / dense feature percentage (firing on > 10% tokens)
    - Firing frequency distribution
    """
    f = dictionary_model.encode(activations)  # (N, d_sae)
    num_tokens = f.shape[0]

    firing_counts = (f > 0).float().sum(dim=0)  # (d_sae,)
    firing_freq = firing_counts / max(1, num_tokens)

    dead_mask = firing_freq < dead_threshold_frac
    dense_mask = firing_freq > 0.10

    return {
        "total_features": dictionary_model.d_sae,
        "dead_features_count": dead_mask.sum().item(),
        "dead_features_pct": (dead_mask.sum().item() / dictionary_model.d_sae) * 100.0,
        "dense_features_count": dense_mask.sum().item(),
        "dense_features_pct": (dense_mask.sum().item() / dictionary_model.d_sae) * 100.0,
        "mean_activation_when_fired": f[f > 0].mean().item() if (f > 0).any() else 0.0,
    }
