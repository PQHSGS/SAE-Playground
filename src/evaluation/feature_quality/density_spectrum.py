from typing import Dict, Any
import torch
from src.core.base_dictionary import BaseDictionary


@torch.no_grad()
def compute_activation_density_spectrum(
    dictionary_model: BaseDictionary,
    activations: torch.Tensor,
) -> Dict[str, Any]:
    """
    Computes global feature firing density spectrum across frequency bins:
    - Dead Latents: Firing frequency < 1e-6
    - Ultra-Rare Latents: Firing frequency in [1e-6, 1e-4)
    - Rare Latents: Firing frequency in [1e-4, 1e-3)
    - Common Latents: Firing frequency in [1e-3, 1e-2)
    - Ultra-Dense / Monolithic Latents: Firing frequency >= 1e-2 (1%)
    - Effective Latent Utilization: Percentage of latents with healthy firing frequency
    """
    device = next(dictionary_model.parameters()).device
    activations = activations.to(device)

    f = dictionary_model.encode(activations)  # (N, d_sae)
    if f.ndim > 2:
        f = f.view(-1, f.shape[-1])
    num_tokens, d_sae = f.shape

    firing_counts = (f > 0).float().sum(dim=0)  # (d_sae,)
    firing_freq = firing_counts / max(1, num_tokens)

    dead_mask = firing_freq < 1e-6
    ultra_rare_mask = (firing_freq >= 1e-6) & (firing_freq < 1e-4)
    rare_mask = (firing_freq >= 1e-4) & (firing_freq < 1e-3)
    common_mask = (firing_freq >= 1e-3) & (firing_freq < 1e-2)
    dense_mask = firing_freq >= 1e-2

    dead_pct = (dead_mask.sum().item() / d_sae) * 100.0
    ultra_rare_pct = (ultra_rare_mask.sum().item() / d_sae) * 100.0
    rare_pct = (rare_mask.sum().item() / d_sae) * 100.0
    common_pct = (common_mask.sum().item() / d_sae) * 100.0
    dense_pct = (dense_mask.sum().item() / d_sae) * 100.0
    healthy_utilization_pct = ((~dead_mask & ~dense_mask).sum().item() / d_sae) * 100.0

    return {
        "total_features": d_sae,
        "dead_features_pct": dead_pct,
        "ultra_rare_features_pct": ultra_rare_pct,
        "rare_features_pct": rare_pct,
        "common_features_pct": common_pct,
        "ultra_dense_features_pct": dense_pct,
        "healthy_utilization_pct": healthy_utilization_pct,
        "mean_fired_magnitude": f[f > 0].mean().item() if (f > 0).any() else 0.0,
    }
