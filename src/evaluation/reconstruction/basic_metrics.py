from typing import Dict, Optional
import torch
from src.core.base_dictionary import BaseDictionary


@torch.no_grad()
def compute_reconstruction_metrics(
    dictionary_model: BaseDictionary,
    activations: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute standard dictionary reconstruction quality metrics:
    - MSE: Mean squared reconstruction error ||x - x_hat||^2
    - NMSE: Normalized MSE = MSE / Var(x)
    - FVE: Fraction of Variance Explained = 1 - NMSE
    - Cosine Similarity: Mean cosine angle between target and reconstruction
    - L0: Average active latents per token
    - L1: Mean L1 activation norm
    """
    device = next(dictionary_model.parameters()).device
    activations = activations.to(device)
    targets = targets.to(device) if targets is not None else activations
    
    out = dictionary_model(activations, target=targets)
    x_hat = out.reconstructed
    f = out.feature_acts

    residual = targets - x_hat
    mse = torch.mean(residual ** 2).item()
    target_var = torch.var(targets).item() + 1e-8
    nmse = mse / target_var
    fve = max(0.0, 1.0 - (torch.var(residual).item() / target_var))

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(
        targets.view(-1, targets.shape[-1]),
        x_hat.view(-1, x_hat.shape[-1]),
        dim=-1
    ).mean().item()

    l0 = (f > 0).float().sum(dim=-1).mean().item()
    l1 = f.sum(dim=-1).mean().item()

    return {
        "mse": mse,
        "nmse": nmse,
        "fraction_variance_explained": fve,
        "cosine_similarity": cos_sim,
        "l0": l0,
        "l1": l1,
    }
