from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from src.core.base_dictionary import BaseDictionary
from src.core.hook_manager import HookManager


@torch.no_grad()
def compute_reconstruction_metrics(
    dictionary_model: BaseDictionary,
    activations: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute standard dictionary reconstruction quality metrics:
    - NMSE (Normalized Mean Squared Error)
    - MSE
    - L0 (average active latents per token)
    - L1 (average L1 activation norm)
    - Explained Variance = 1 - Var(x - x_hat) / Var(x)
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
    explained_var = 1.0 - (torch.var(residual).item() / target_var)

    l0 = (f > 0).float().sum(dim=-1).mean().item()
    l1 = f.sum(dim=-1).mean().item()

    return {
        "mse": mse,
        "nmse": nmse,
        "explained_variance": max(0.0, explained_var),
        "l0": l0,
        "l1": l1,
    }


@torch.no_grad()
def compute_ce_loss_recovery(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dictionary_model: BaseDictionary,
    hook_point: str,
    test_texts: list[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    context_length: int = 1024,
) -> Dict[str, float]:
    """
    Measures Cross-Entropy Loss Recovery when replacing model activations at `hook_point`
    with dictionary reconstructions x_hat:
    CE_loss_recovered = (CE_zero_ablation - CE_reconstructed) / (CE_zero_ablation - CE_original)
    """
    hook_manager = HookManager(model)
    encoding = tokenizer(
        test_texts,
        truncation=True,
        max_length=context_length,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 1. Original model cross-entropy loss
    orig_out = model(**encoding, labels=encoding["input_ids"])
    orig_loss = orig_out.loss.item()

    # 2. Zero-ablation baseline (setting hooked activations to 0)
    def zero_ablation_hook(tensor):
        return torch.zeros_like(tensor)

    hook_manager.register_intervention_hook(hook_point, zero_ablation_hook)
    zero_out = model(**encoding, labels=encoding["input_ids"])
    zero_loss = zero_out.loss.item()
    hook_manager.remove_hooks()

    # 3. Reconstruction intervention (splicing dictionary x_hat)
    def reconstruction_hook(tensor):
        shape = tensor.shape
        flat = tensor.view(-1, shape[-1])
        with torch.no_grad():
            f = dictionary_model.encode(flat)
            x_hat = dictionary_model.decode(f)
        return x_hat.view(shape)

    hook_manager.register_intervention_hook(hook_point, reconstruction_hook)
    recon_out = model(**encoding, labels=encoding["input_ids"])
    recon_loss = recon_out.loss.item()
    hook_manager.remove_hooks()

    # Recovery ratio calculation
    denom = max(1e-6, zero_loss - orig_loss)
    ce_recovery = (zero_loss - recon_loss) / denom

    return {
        "ce_original": orig_loss,
        "ce_zero_ablation": zero_loss,
        "ce_reconstructed": recon_loss,
        "ce_loss_recovery": ce_recovery,
    }
