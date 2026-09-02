import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.core.base_dictionary import BaseDictionary
from src.core.multi_dictionary import MultiLayerDictionary
from src.core.hook_manager import HookManager
from src.utils.logging import setup_logger

logger = setup_logger("circuits.transcoder")


@dataclass
class TranscoderEdge:
    source_layer: str
    source_feature: int
    target_layer: str
    target_feature: int
    attribution_score: float


@dataclass
class TranscoderCircuitResult:
    faithfulness_loss_recovery: float
    baseline_loss: float
    replaced_loss: float
    top_edges: List[TranscoderEdge]
    layer_feature_attributions: Dict[str, Dict[int, float]]


class TranscoderCircuitGraph:
    """
    Transcoder Circuit Discovery & Sublayer Replacement Engine.
    
    Implements mechanistic interpretability circuit discovery via Transcoders:
    1. Sublayer Replacement: Replaces non-linear MLP modules with linear feature Transcoders.
    2. Faithfulness Evaluation: Measures CE loss recovery under full or partial transcoder replacement.
    3. Cross-Layer Feature Attribution: Traces causal computational paths from Transcoder_i to Transcoder_j.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        transcoders: Union[MultiLayerDictionary, Dict[str, BaseDictionary]],
        hook_pairs: Dict[str, str],  # Maps input_hook (e.g. "model.layers.0.mlp") -> output_hook (e.g. "model.layers.0.mlp.down_proj")
    ):
        self.model = model
        self.tokenizer = tokenizer
        if isinstance(transcoders, dict):
            self.transcoders = MultiLayerDictionary(transcoders)
        else:
            self.transcoders = transcoders
        self.hook_pairs = hook_pairs
        self.hook_manager = HookManager(model)

    def evaluate_replacement_faithfulness(
        self,
        eval_texts: List[str],
        max_length: int = 128,
    ) -> Dict[str, float]:
        """
        Measures the model's loss recovery when MLP sublayers are replaced by Transcoders.
        Faithfulness Score = 1.0 - (Replaced_Loss - Baseline_Loss) / Baseline_Loss.
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        baseline_losses = []
        replaced_losses = []

        loss_fn = nn.CrossEntropyLoss()

        for text in eval_texts:
            inputs = self.tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True).to(device)
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            # 1. Baseline Model Loss
            with torch.no_grad():
                out = self.model(**inputs)
                shift_logits = out.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                base_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()
                baseline_losses.append(base_loss)

            # 2. Replaced Model Loss (MLP replaced by Transcoder)
            handles = []
            layer_inputs = {}

            def make_input_hook(hp_in: str):
                def hook(module, inp, out):
                    layer_inputs[hp_in] = inp[0] if isinstance(inp, tuple) else inp
                return hook

            def make_output_hook(hp_in: str):
                def hook(module, inp, out):
                    if hp_in in layer_inputs:
                        x_in = layer_inputs[hp_in]
                        sae = self.transcoders.get_dictionary(hp_in)
                        w_ref = sae.get_decoder_weights()
                        x_in = x_in.to(device=w_ref.device, dtype=w_ref.dtype)
                        f = sae.encode(x_in)
                        if isinstance(out, tuple):
                            target_dtype = out[0].dtype
                            y_hat = sae.decode(f).to(device=out[0].device, dtype=target_dtype)
                            return (y_hat,) + out[1:]
                        else:
                            y_hat = sae.decode(f).to(device=out.device, dtype=out.dtype)
                            return y_hat
                    return out
                return hook

            for hp_in, hp_out in self.hook_pairs.items():
                mod_in = self.hook_manager.get_submodule(hp_in)
                mod_out = self.hook_manager.get_submodule(hp_out)
                handles.append(mod_in.register_forward_hook(make_input_hook(hp_in)))
                handles.append(mod_out.register_forward_hook(make_output_hook(hp_in)))

            try:
                with torch.no_grad():
                    rep_out = self.model(**inputs)
                    shift_rep_logits = rep_out.logits[:, :-1, :].contiguous()
                    rep_loss = loss_fn(shift_rep_logits.view(-1, shift_rep_logits.size(-1)), shift_labels.view(-1)).item()
                    replaced_losses.append(rep_loss)
            finally:
                for h in handles:
                    h.remove()

        mean_base = sum(baseline_losses) / max(1, len(baseline_losses))
        mean_rep = sum(replaced_losses) / max(1, len(replaced_losses))
        loss_diff = mean_rep - mean_base
        faithfulness = max(0.0, 1.0 - (loss_diff / max(1e-8, mean_base)))

        logger.info(f"Replacement Faithfulness: {faithfulness * 100:.2f}% (Base Loss: {mean_base:.4f} -> Replaced: {mean_rep:.4f})")
        return {
            "faithfulness": faithfulness,
            "baseline_loss": mean_base,
            "replaced_loss": mean_rep,
            "delta_loss": loss_diff,
        }

    def compute_transcoder_attributions(
        self,
        clean_text: str,
        corrupted_text: str,
        target_token_id: int,
        top_k_edges: int = 30,
    ) -> TranscoderCircuitResult:
        """
        Computes causal edge attributions between Transcoders across layers
        using first-order Taylor expansion on contrastive inputs.
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        input_hooks = list(self.hook_pairs.keys())
        self.hook_manager.register_forward_hooks(input_hooks)

        # 1. Clean forward pass
        clean_inputs = self.tokenizer(clean_text, return_tensors="pt").to(device)
        clean_out = self.model(**clean_inputs)
        target_logit = clean_out.logits[0, -1, target_token_id]

        # Get clean activations and compute gradients
        clean_acts = {hp: self.hook_manager.activations[hp] for hp in input_hooks}
        grads = torch.autograd.grad(
            target_logit,
            list(clean_acts.values()),
            retain_graph=False,
            create_graph=False,
        )
        grad_dict = {hp: g[0, -1].detach() for hp, g in zip(input_hooks, grads)}
        self.hook_manager.remove_hooks()

        # 2. Corrupted forward pass
        with torch.no_grad():
            corr_inputs = self.tokenizer(corrupted_text, return_tensors="pt").to(device)
            self.hook_manager.register_forward_hooks(input_hooks)
            _ = self.model(**corr_inputs)
            corr_acts = {hp: self.hook_manager.activations[hp][0, -1].detach() for hp in input_hooks}
            self.hook_manager.remove_hooks()

        # 3. Compute per-feature and cross-layer edge attributions
        top_edges: List[TranscoderEdge] = []
        layer_attributions: Dict[str, Dict[int, float]] = {}

        for i, src_hp in enumerate(input_hooks):
            src_sae = self.transcoders.get_dictionary(src_hp)
            w_dec = src_sae.get_decoder_weights().to(device)
            f_clean = src_sae.encode(clean_acts[src_hp][0, -1].detach().to(device=w_dec.device, dtype=w_dec.dtype))
            f_corr = src_sae.encode(corr_acts[src_hp].to(device=w_dec.device, dtype=w_dec.dtype))
            delta_f = f_clean - f_corr

            grad_src = grad_dict[src_hp].to(device=w_dec.device, dtype=w_dec.dtype)
            grad_f = grad_src @ w_dec.T
            node_attr = (delta_f * grad_f).abs()

            top_k_indices = torch.topk(node_attr, k=min(10, src_sae.d_sae)).indices.tolist()
            layer_attributions[src_hp] = {idx: node_attr[idx].item() for idx in top_k_indices}

            for tgt_hp in input_hooks[i + 1:]:
                tgt_sae = self.transcoders.get_dictionary(tgt_hp)
                tgt_w_dec = tgt_sae.get_decoder_weights().to(device)
                grad_tgt = grad_dict[tgt_hp].to(device=tgt_w_dec.device, dtype=tgt_w_dec.dtype)
                grad_tgt_f = grad_tgt @ tgt_w_dec.T

                edge_matrix = torch.outer(delta_f.abs()[:50], grad_tgt_f.abs()[:50])
                flat_top = torch.topk(edge_matrix.view(-1), k=min(3, edge_matrix.numel()))
                
                for val, idx in zip(flat_top.values, flat_top.indices):
                    s_idx = idx.item() // 50
                    t_idx = idx.item() % 50
                    top_edges.append(TranscoderEdge(
                        source_layer=src_hp,
                        source_feature=s_idx,
                        target_layer=tgt_hp,
                        target_feature=t_idx,
                        attribution_score=val.item(),
                    ))

        top_edges.sort(key=lambda x: x.attribution_score, reverse=True)
        top_edges = top_edges[:top_k_edges]

        faith_res = self.evaluate_replacement_faithfulness([clean_text])

        return TranscoderCircuitResult(
            faithfulness_loss_recovery=faith_res["faithfulness"],
            baseline_loss=faith_res["baseline_loss"],
            replaced_loss=faith_res["replaced_loss"],
            top_edges=top_edges,
            layer_feature_attributions=layer_attributions,
        )

    def export_graph_json(self, result: TranscoderCircuitResult, save_path: Optional[str] = None) -> Dict:
        """Exports circuit nodes and causal edges in JSON-serializable graph format."""
        nodes = []
        for layer, feats in result.layer_feature_attributions.items():
            for feat_id, score in feats.items():
                nodes.append({
                    "id": f"{layer}::feat_{feat_id}",
                    "layer": layer,
                    "feature_id": feat_id,
                    "attribution": score,
                })

        edges = [
            {
                "source": f"{e.source_layer}::feat_{e.source_feature}",
                "target": f"{e.target_layer}::feat_{e.target_feature}",
                "score": e.attribution_score,
            }
            for e in result.top_edges
        ]

        graph_data = {
            "faithfulness": result.faithfulness_loss_recovery,
            "baseline_loss": result.baseline_loss,
            "replaced_loss": result.replaced_loss,
            "nodes": nodes,
            "edges": edges,
        }

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, indent=2)
            logger.info(f"Exported Transcoder circuit graph to: {save_path}")

        return graph_data
