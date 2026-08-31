from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from src.core.base_dictionary import BaseDictionary
from src.core.hook_manager import HookManager
from src.utils.logging import setup_logger

logger = setup_logger("circuits.eap")


@dataclass
class CircuitAttributionResult:
    source_attributions: torch.Tensor       # Shape: (d_sae_source,)
    target_attributions: torch.Tensor       # Shape: (d_sae_target,)
    edge_attributions: torch.Tensor         # Shape: (d_sae_source, d_sae_target)
    top_source_features: List[Tuple[int, float]]
    top_target_features: List[Tuple[int, float]]
    top_edges: List[Tuple[int, int, float]]


class EdgeAttributionPatching:
    """
    Edge Attribution Patching for Sparse Autoencoders (EAP-SAE).
    Identifies causal sub-circuits connecting features across transformer layers
    using first-order Taylor expansion on contrastive clean vs corrupted inputs.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        source_sae: BaseDictionary,
        target_sae: BaseDictionary,
        source_hook: str,
        target_hook: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.source_sae = source_sae
        self.target_sae = target_sae
        self.source_hook = source_hook
        self.target_hook = target_hook
        self.hook_manager = HookManager(model)

    def compute_circuit_attributions(
        self,
        clean_text: str,
        corrupted_text: str,
        target_token_id: int,
        top_k_edges: int = 50,
    ) -> CircuitAttributionResult:
        """
        Computes node and edge attributions between source and target SAE features.
        """
        logger.info(f"Computing EAP-SAE between '{self.source_hook}' and '{self.target_hook}'")
        device = next(self.model.parameters()).device

        # 1. Clean forward pass with autograd gradients enabled
        clean_inputs = self.tokenizer(clean_text, return_tensors="pt").to(device)
        self.hook_manager.register_forward_hooks([self.source_hook, self.target_hook])

        clean_out = self.model(**clean_inputs)
        clean_src_act = self.hook_manager.activations[self.source_hook]
        clean_tgt_act = self.hook_manager.activations[self.target_hook]

        # Logit target loss: - logit(target_token)
        target_logit = clean_out.logits[0, -1, target_token_id]

        # Compute gradients with respect to source and target activations
        grads = torch.autograd.grad(
            target_logit,
            [clean_src_act, clean_tgt_act],
            retain_graph=False,
            create_graph=False,
        )
        grad_src_act = grads[0][0, -1].detach()  # (d_in_src,)
        grad_tgt_act = grads[1][0, -1].detach()  # (d_in_tgt,)
        self.hook_manager.remove_hooks()

        # 2. Corrupted forward pass under torch.no_grad()
        with torch.no_grad():
            corr_inputs = self.tokenizer(corrupted_text, return_tensors="pt").to(device)
            self.hook_manager.register_forward_hooks([self.source_hook, self.target_hook])
            _ = self.model(**corr_inputs)
            corr_src_act = self.hook_manager.activations[self.source_hook][0, -1].detach()
            self.hook_manager.remove_hooks()

            # Encode features on clean vs corrupted activations
            f_src_clean = self.source_sae.encode(clean_src_act[0, -1].detach()).float()  # (d_sae_src,)
            f_src_corr = self.source_sae.encode(corr_src_act).float()                    # (d_sae_src,)
            delta_f_src = f_src_clean - f_src_corr                                       # (d_sae_src,)

            f_tgt_clean = self.target_sae.encode(clean_tgt_act[0, -1].detach()).float()  # (d_sae_tgt,)

            # Project gradients to feature space via decoder matrices
            w_dec_src = self.source_sae.get_decoder_weights().float().to(device)         # (d_sae_src, d_in_src)
            w_dec_tgt = self.target_sae.get_decoder_weights().float().to(device)         # (d_sae_tgt, d_in_tgt)

            grad_f_src = torch.matmul(w_dec_src, grad_src_act.float())                   # (d_sae_src,)
            grad_f_tgt = torch.matmul(w_dec_tgt, grad_tgt_act.float())                   # (d_sae_tgt,)

            # Node attributions
            attr_src = torch.abs(delta_f_src * grad_f_src)
            attr_tgt = torch.abs(f_tgt_clean * grad_f_tgt)

            # Direct feature-to-feature edge attribution (first-order direct interaction)
            # Alignment between source decoder vector and target gradient
            src_tgt_alignment = torch.matmul(w_dec_src, grad_tgt_act.float())            # (d_sae_src,)
            edge_attr = torch.abs(delta_f_src.unsqueeze(1) * src_tgt_alignment.unsqueeze(1) * grad_f_tgt.unsqueeze(0))

        # Top features & edges extraction
        top_src_vals, top_src_idx = torch.topk(attr_src, k=min(20, attr_src.shape[0]))
        top_tgt_vals, top_tgt_idx = torch.topk(attr_tgt, k=min(20, attr_tgt.shape[0]))

        top_src_list = [(idx.item(), val.item()) for idx, val in zip(top_src_idx, top_src_vals)]
        top_tgt_list = [(idx.item(), val.item()) for idx, val in zip(top_tgt_idx, top_tgt_vals)]

        # Top edges
        flat_edges = edge_attr.view(-1)
        top_edge_vals, top_edge_idx = torch.topk(flat_edges, k=min(top_k_edges, flat_edges.shape[0]))
        top_edges = []
        d_tgt = edge_attr.shape[1]
        for val, idx in zip(top_edge_vals, top_edge_idx):
            src_i = (idx // d_tgt).item()
            tgt_j = (idx % d_tgt).item()
            top_edges.append((src_i, tgt_j, val.item()))

        logger.info(f"EAP-SAE completed: identified {len(top_edges)} candidate circuit edges")
        return CircuitAttributionResult(
            source_attributions=attr_src,
            target_attributions=attr_tgt,
            edge_attributions=edge_attr,
            top_source_features=top_src_list,
            top_target_features=top_tgt_list,
            top_edges=top_edges,
        )
