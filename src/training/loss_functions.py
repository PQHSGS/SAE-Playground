from typing import Optional
import torch
import torch.nn as nn


def normalized_mse_loss(
    x_hat: torch.Tensor,
    target: torch.Tensor,
    running_mean: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Normalized MSE Loss: ||target - x_hat||^2 / ||target - E[target]||^2
    Provides scale-independent reconstruction error across layers and models.
    """
    mse = torch.mean((target - x_hat) ** 2)
    if running_mean is not None:
        variance = torch.mean((target - running_mean) ** 2) + eps
    else:
        variance = torch.var(target) + eps
    return mse / variance


def compute_ghost_gradients_loss(
    residual: torch.Tensor,
    pre_acts: torch.Tensor,
    w_dec: torch.Tensor,
    dead_mask: torch.Tensor,
    ghost_grad_coeff: float = 0.1,
    topk_dead: int = 32
) -> torch.Tensor:
    """
    Ghost Gradients (OpenAI / DeepMind):
    Passes reconstruction error gradients to inactive latents without altering forward outputs.
    """
    if not dead_mask.any() or ghost_grad_coeff <= 0:
        return torch.tensor(0.0, device=residual.device)

    dead_acts = pre_acts * dead_mask.float()
    k = min(topk_dead, int(dead_mask.sum().item()))
    if k == 0:
        return torch.tensor(0.0, device=residual.device)

    dead_val, dead_idx = torch.topk(dead_acts, k=k, dim=-1)
    ghost_f = torch.zeros_like(dead_acts)
    ghost_f.scatter_(dim=-1, index=dead_idx, src=dead_val)

    if w_dec.ndim == 2:
        ghost_recon = torch.matmul(ghost_f, w_dec)
    elif w_dec.ndim == 3:  # Crosscoder
        ghost_recon = torch.einsum("...d,ldi->...li", ghost_f, w_dec)

    return ghost_grad_coeff * nn.functional.mse_loss(ghost_recon, residual.detach())
