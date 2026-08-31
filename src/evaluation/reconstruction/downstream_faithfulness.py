from typing import Dict, List
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from src.core.base_dictionary import BaseDictionary
from src.core.hook_manager import HookManager


@torch.no_grad()
def compute_ce_loss_recovery(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dictionary_model: BaseDictionary,
    hook_point: str,
    test_texts: List[str],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    context_length: int = 1024,
) -> Dict[str, float]:
    """
    Measures downstream LM faithfulness when replacing internal activations at `hook_point`
    with dictionary reconstructions x_hat:
    - CE Loss Recovery: (CE_zero - CE_recon) / (CE_zero - CE_orig)
    - KL Divergence: Mean D_KL(P_clean || P_recon) across output token logits
    - Top-1 Agreement: Percentage of tokens where argmax(P_recon) == argmax(P_clean)
    """
    hook_manager = HookManager(model)
    encoding = tokenizer(
        test_texts,
        truncation=True,
        max_length=context_length,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 1. Clean original forward pass
    orig_out = model(**encoding, labels=encoding["input_ids"])
    orig_loss = orig_out.loss.item()
    orig_logits = orig_out.logits

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
        f = dictionary_model.encode(flat)
        x_hat = dictionary_model.decode(f)
        return x_hat.view(shape)

    hook_manager.register_intervention_hook(hook_point, reconstruction_hook)
    recon_out = model(**encoding, labels=encoding["input_ids"])
    recon_loss = recon_out.loss.item()
    recon_logits = recon_out.logits
    hook_manager.remove_hooks()

    # 4. Compute Faithfulness Metrics
    denom = max(1e-6, zero_loss - orig_loss)
    ce_recovery = (zero_loss - recon_loss) / denom

    # KL Divergence between clean and reconstructed output distributions
    clean_probs = F.log_softmax(orig_logits, dim=-1)
    recon_probs = F.softmax(recon_logits, dim=-1)
    kl_div = F.kl_div(clean_probs, recon_probs, reduction="batchmean", log_target=False).item()

    # Top-1 token prediction accuracy preservation
    clean_preds = orig_logits.argmax(dim=-1)
    recon_preds = recon_logits.argmax(dim=-1)
    top1_agreement = (clean_preds == recon_preds).float().mean().item() * 100.0

    return {
        "ce_loss_original": orig_loss,
        "ce_loss_zero_ablation": zero_loss,
        "ce_loss_reconstructed": recon_loss,
        "ce_loss_recovered": float(ce_recovery),
        "kl_divergence": kl_div,
        "top1_token_agreement_pct": top1_agreement,
    }
