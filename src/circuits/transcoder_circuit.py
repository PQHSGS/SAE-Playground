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
                raw_logits = out.logits if hasattr(out, "logits") else out
                shift_logits = raw_logits[:, :-1, :].contiguous()
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
                    raw_rep = rep_out.logits if hasattr(rep_out, "logits") else rep_out
                    shift_rep_logits = raw_rep[:, :-1, :].contiguous()
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
        raw_clean = clean_out.logits if hasattr(clean_out, "logits") else clean_out
        target_logit = raw_clean[0, -1, target_token_id]

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


@dataclass
class AttributionNode:
    id: str
    node_type: str  # "input_token" | "feature" | "logit"
    layer: str
    layer_idx: int
    pos: int
    feature_id: Optional[int]
    label: str
    title: str
    explanation: str
    activation: float
    logit_influence: float
    promoted_tokens: List[str]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "node_type": self.node_type,
            "layer": self.layer,
            "layer_idx": self.layer_idx,
            "pos": self.pos,
            "feature_id": self.feature_id,
            "label": self.label,
            "title": self.title,
            "explanation": self.explanation,
            "activation": float(self.activation),
            "logit_influence": float(self.logit_influence),
            "promoted_tokens": self.promoted_tokens,
        }


@dataclass
class AttributionEdge:
    source: str
    target: str
    weight: float

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "weight": float(self.weight),
        }


class AnthropicAttributionGraphEngine:
    """
    Anthropic Transformer Circuits Attribution Graph Engine.
    Reference: 'Circuit Tracing: Revealing Computational Graphs in Language Models'
               (Ameisen et al., Anthropic, March 2025).

    Methodology:
    1. Replacement Model with Cross-Layer / Inter-Layer Transcoders.
    2. Linear feature-feature attribution with frozen attention & RMSNorm.
    3. Pairwise Virtual Weights: V_{st} = <W_dec^s, W_enc^t>.
    4. Neumann Series Indirect Influence: B = (I - A)^{-1} - I.
    5. Cumulative influence pruning (threshold tau=0.80) & edge filtering.
    6. Automatic semantic labeling via feature auto-interpretation metadata.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        transcoders: Union[MultiLayerDictionary, Dict[str, BaseDictionary]],
        feature_metadata: Optional[Dict[str, Dict]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if isinstance(transcoders, dict):
            self.transcoders = MultiLayerDictionary(transcoders)
        else:
            self.transcoders = transcoders
        self.feature_metadata = feature_metadata or {}
        self.hook_manager = HookManager(model)

    def get_feature_meta(self, layer: str, feat_id: int) -> Dict:
        candidates = [
            layer,
            layer.replace(".", "_"),
            layer.replace("model.layers.", ""),
            layer.replace("transformer.", ""),
        ]
        layer_dict = {}
        for cand in candidates:
            if cand in self.feature_metadata:
                layer_dict = self.feature_metadata[cand]
                break
        return layer_dict.get(str(feat_id), layer_dict.get(feat_id, {}))

    def trace_graph(
        self,
        prompt: str,
        target_token_id: Optional[int] = None,
        pruning_threshold: float = 0.80,
        max_nodes: int = 40,
        max_edges: int = 50,
        top_k_features_per_layer: int = 4,
    ) -> Dict:
        device = next(self.model.parameters()).device
        self.model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0].tolist()
        seq_len = len(input_ids)
        token_strs = [self.tokenizer.decode([tid]) for tid in input_ids]

        # Identify all transcoder hook points
        if hasattr(self.transcoders, "hook_point_map"):
            hook_points = list(self.transcoders.hook_point_map.values())
        else:
            hook_points = list(self.transcoders.dictionaries.keys())

        # Sort hook points by layer index
        def extract_layer_idx(hp: str) -> int:
            nums = [int(s) for s in hp.replace("_", ".").split(".") if s.isdigit()]
            return nums[0] if nums else 0

        hook_points = sorted(hook_points, key=extract_layer_idx)

        # 1. Forward Pass to gather activations
        self.hook_manager.register_forward_hooks(hook_points)
        out = self.model(**inputs)
        raw_logits = out.logits if hasattr(out, "logits") else out
        logits = raw_logits[0, -1]
        probs = torch.softmax(logits, dim=-1)

        if target_token_id is None:
            target_token_id = int(torch.argmax(logits).item())
        target_token_str = self.tokenizer.decode([target_token_id])
        target_prob = float(probs[target_token_id].item())

        target_logit = logits[target_token_id]

        # Capture layer activations
        acts_dict = {hp: self.hook_manager.activations[hp] for hp in hook_points}

        # Compute gradients with respect to layer activations
        grads = torch.autograd.grad(
            target_logit,
            list(acts_dict.values()),
            retain_graph=False,
            create_graph=False,
        )
        grad_dict = {hp: g[0].detach() for hp, g in zip(hook_points, grads)}  # (seq_len, d_in)
        self.hook_manager.remove_hooks()

        # 2. Extract Candidate Active Features
        candidate_nodes: Dict[str, AttributionNode] = {}
        features_by_layer: Dict[str, List[Dict]] = {hp: [] for hp in hook_points}

        # Create input token nodes
        for pos, t_str in enumerate(token_strs):
            tok_id = f"tok_{pos}"
            candidate_nodes[tok_id] = AttributionNode(
                id=tok_id,
                node_type="input_token",
                layer="input",
                layer_idx=-1,
                pos=pos,
                feature_id=None,
                label=t_str,
                title=f"Token [{t_str.strip()}]",
                explanation=f"Prompt token '{t_str}' at position {pos}",
                activation=1.0,
                logit_influence=0.0,
                promoted_tokens=[],
            )

        # Create candidate transcoder feature nodes
        for lyr_idx, hp in enumerate(hook_points):
            sae = self.transcoders.get_dictionary(hp)
            x_act = acts_dict[hp][0].detach()  # (seq_len, d_in)
            scale = (sae.d_in ** 0.5) / (x_act.norm(dim=-1, keepdim=True) + 1e-8)
            normed_act = (x_act * scale).to(device=sae.get_decoder_weights().device, dtype=sae.get_decoder_weights().dtype)
            
            with torch.no_grad():
                f_acts = sae.encode(normed_act)  # (seq_len, d_sae)

            w_dec = sae.get_decoder_weights().to(device)
            g_layer = grad_dict[hp].to(device=w_dec.device, dtype=w_dec.dtype)  # (seq_len, d_in)
            g_feats = g_layer @ w_dec.T  # (seq_len, d_sae)

            # Direct attribution to target logit
            direct_attr = (f_acts.to(device) * g_feats).abs()  # (seq_len, d_sae)

            # Select top features across sequence positions for this layer
            flat_attr = direct_attr.view(-1)
            top_k_val, top_k_idx = torch.topk(flat_attr, k=min(top_k_features_per_layer, flat_attr.numel()))

            for val, flat_i in zip(top_k_val.tolist(), top_k_idx.tolist()):
                if val <= 1e-5:
                    continue
                pos = flat_i // sae.d_sae
                feat_idx = flat_i % sae.d_sae
                act_val = float(f_acts[pos, feat_idx].item())

                fmeta = self.get_feature_meta(hp, feat_idx)
                node_id = f"{hp}::pos_{pos}::feat_{feat_idx}"
                title = fmeta.get("title", f"Feature #{feat_idx}")
                explanation = fmeta.get("explanation", "")
                promoted = fmeta.get("top_promoted_tokens", [])

                node = AttributionNode(
                    id=node_id,
                    node_type="feature",
                    layer=hp,
                    layer_idx=lyr_idx,
                    pos=int(pos),
                    feature_id=int(feat_idx),
                    label=f"F#{feat_idx}",
                    title=title,
                    explanation=explanation,
                    activation=act_val,
                    logit_influence=float(val),
                    promoted_tokens=promoted,
                )
                candidate_nodes[node_id] = node
                features_by_layer[hp].append({
                    "id": node_id,
                    "pos": int(pos),
                    "feat_idx": int(feat_idx),
                    "act": act_val,
                    "grad": float(g_feats[pos, feat_idx].item()),
                    "direct_attr": float(val),
                })

        # Create output logit node
        logit_node_id = f"logit::{target_token_str.strip()}"
        candidate_nodes[logit_node_id] = AttributionNode(
            id=logit_node_id,
            node_type="logit",
            layer="output",
            layer_idx=len(hook_points),
            pos=seq_len - 1,
            feature_id=None,
            label=f"Logit: {target_token_str.strip()}",
            title=f"Target Token '{target_token_str.strip()}'",
            explanation=f"Final prediction probability: {target_prob * 100:.1f}%",
            activation=target_prob,
            logit_influence=1.0,
            promoted_tokens=[target_token_str.strip()],
        )

        # 3. Compute Direct Pairwise Edges
        raw_edges: List[AttributionEdge] = []

        # Token -> First Layer Features edges
        if hook_points:
            first_hp = hook_points[0]
            for f_info in features_by_layer[first_hp]:
                f_pos = f_info["pos"]
                tok_id = f"tok_{f_pos}"
                raw_edges.append(AttributionEdge(
                    source=tok_id,
                    target=f_info["id"],
                    weight=f_info["direct_attr"],
                ))

        # Feature -> Feature cross-layer edges (Taylor virtual weights)
        for i, src_hp in enumerate(hook_points[:-1]):
            src_sae = self.transcoders.get_dictionary(src_hp)
            src_w_dec = src_sae.get_decoder_weights()
            for tgt_hp in hook_points[i + 1: i + 4]:  # Restrict to nearby downstream layers
                tgt_sae = self.transcoders.get_dictionary(tgt_hp)
                tgt_w_enc = tgt_sae.w_enc.T if hasattr(tgt_sae, "w_enc") else tgt_sae.get_decoder_weights()

                for s_item in features_by_layer[src_hp]:
                    s_w = src_w_dec[s_item["feat_idx"]].to(device, dtype=torch.float32)
                    for t_item in features_by_layer[tgt_hp]:
                        t_w = tgt_w_enc[t_item["feat_idx"]].to(device, dtype=torch.float32)
                        # Virtual weight V_st = <W_dec^s, W_enc^t>
                        v_st = float(torch.dot(s_w, t_w).item())
                        edge_weight = abs(s_item["act"] * v_st * t_item["grad"])
                        if edge_weight > 1e-4:
                            raw_edges.append(AttributionEdge(
                                source=s_item["id"],
                                target=t_item["id"],
                                weight=edge_weight,
                            ))

        # Later Feature -> Output Logit edges
        for hp in hook_points[-4:]:
            for f_item in features_by_layer[hp]:
                raw_edges.append(AttributionEdge(
                    source=f_item["id"],
                    target=logit_node_id,
                    weight=f_item["direct_attr"],
                ))

        # 4. Indirect Influence Matrix & Neumann Series: B = (I - A)^{-1} - I
        all_node_keys = list(candidate_nodes.keys())
        node_idx_map = {k: i for i, k in enumerate(all_node_keys)}
        N = len(all_node_keys)

        adj_matrix = torch.zeros((N, N), dtype=torch.float32)
        for e in raw_edges:
            if e.source in node_idx_map and e.target in node_idx_map:
                s_i = node_idx_map[e.source]
                t_i = node_idx_map[e.target]
                adj_matrix[t_i, s_i] = max(adj_matrix[t_i, s_i].item(), abs(e.weight))

        # Column-normalize incoming edges to sum to 1
        col_sums = adj_matrix.sum(dim=1, keepdim=True)
        col_sums[col_sums == 0] = 1.0
        norm_adj = adj_matrix / col_sums

        try:
            # Neumann Series solution: (I - norm_adj)^{-1} - I
            I = torch.eye(N)
            B = torch.inverse(I - 0.95 * norm_adj) - I
            logit_i = node_idx_map[logit_node_id]
            influence_vec = B[logit_i].tolist()
            for k, inf_val in zip(all_node_keys, influence_vec):
                candidate_nodes[k].logit_influence = max(candidate_nodes[k].logit_influence, float(inf_val))
        except Exception:
            # Fallback to direct attribution ranking
            pass

        # 5. Graph Pruning (Anthropic cumulative threshold)
        # Keep tokens and logit unconditionally
        kept_node_ids = set()
        for k, n in candidate_nodes.items():
            if n.node_type in ("input_token", "logit"):
                kept_node_ids.add(k)

        # Rank feature nodes by logit influence
        feat_nodes = [n for n in candidate_nodes.values() if n.node_type == "feature"]
        feat_nodes.sort(key=lambda x: x.logit_influence, reverse=True)

        total_feat_inf = sum(n.logit_influence for n in feat_nodes) + 1e-8
        cum_inf = 0.0
        for n in feat_nodes:
            kept_node_ids.add(n.id)
            cum_inf += n.logit_influence
            if cum_inf / total_feat_inf >= pruning_threshold or len(kept_node_ids) >= max_nodes:
                break

        # Filter edges to kept nodes
        filtered_edges = [
            e for e in raw_edges
            if e.source in kept_node_ids and e.target in kept_node_ids
        ]
        filtered_edges.sort(key=lambda x: x.weight, reverse=True)
        filtered_edges = filtered_edges[:max_edges]

        # Prune unreferenced feature nodes
        connected_ids = set()
        for e in filtered_edges:
            connected_ids.add(e.source)
            connected_ids.add(e.target)
        # Keep all tokens and logit
        for k, n in candidate_nodes.items():
            if n.node_type in ("input_token", "logit"):
                connected_ids.add(k)

        final_nodes = [candidate_nodes[k].to_dict() for k in kept_node_ids if k in connected_ids]
        final_edges = [e.to_dict() for e in filtered_edges]

        top_cand_vals, top_cand_ids = torch.topk(probs, k=min(6, probs.shape[-1]))
        candidates = [
            {"token": self.tokenizer.decode([cid]), "id": int(cid), "prob": float(cp)}
            for cp, cid in zip(top_cand_vals.tolist(), top_cand_ids.tolist())
        ]

        return {
            "prompt": prompt,
            "target_token": target_token_str,
            "target_token_id": int(target_token_id),
            "target_prob": target_prob,
            "candidates": candidates,
            "pruning_threshold": pruning_threshold,
            "nodes": final_nodes,
            "edges": final_edges,
            "metrics": {
                "total_candidate_nodes": N,
                "pruned_nodes": len(final_nodes),
                "pruned_edges": len(final_edges),
                "completeness_score": min(1.0, cum_inf / total_feat_inf),
            }
        }

