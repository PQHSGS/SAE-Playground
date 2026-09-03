import json
import os
from typing import Dict, List, Optional
import torch
from aiohttp import web

from src.utils.hf_helpers import compute_direct_logit_attribution
from src.circuits.steering import FeatureSteeringEngine
from src.core.hook_manager import HookManager
from src.architectures.registry import get_dictionary_cls


class DashboardState:
    model = None
    tokenizer = None
    active_hook_point: str = ""
    checkpoint_dir: str = ""
    available_layers: Dict[str, str] = {}  # {hook_point: subfolder_path}
    loaded_dictionaries: Dict[str, torch.nn.Module] = {}
    unembedding_weights: Optional[torch.Tensor] = None
    feature_metadata: Dict[str, Dict[int, Dict]] = {}  # {hook_point: {feat_id: meta}}
    attribution_engine = None
    sample_contexts: List[str] = [
        "The Eiffel Tower in Paris, France stands on the Champ de Mars near the Seine river.",
        "Quantum mechanics reveals that particles exist in probabilistic superposition wavefunctions.",
        "In deep learning, transformer self-attention enables global contextual token representations.",
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])",
        "The Supreme Court declared the federal law unconstitutional under the Fourteenth Amendment.",
        "Photosynthesis converts carbon dioxide and sunlight into glucose and oxygen molecules in chloroplasts.",
        "The economic inflation rate rose sharply due to supply chain bottlenecks and monetary policy changes.",
        "Machine learning algorithms optimize loss functions using stochastic gradient descent and backpropagation."
    ]

    @property
    def dictionary_model(self) -> Optional[torch.nn.Module]:
        return self.loaded_dictionaries.get(self.active_hook_point)


state = DashboardState()


def get_or_load_dictionary(layer: str) -> Optional[torch.nn.Module]:
    """Lazy-loads and caches the dictionary module for a requested hook point."""
    if layer in state.loaded_dictionaries:
        return state.loaded_dictionaries[layer]

    if layer not in state.available_layers:
        return None

    subfolder = state.available_layers[layer]
    cfg_path = os.path.join(subfolder, "config.json")
    if not os.path.exists(cfg_path):
        return None

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cls = get_dictionary_cls(cfg.get("class_name", "TopKSAE"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dict_model = cls.from_pretrained(subfolder, device=device)
    state.loaded_dictionaries[layer] = dict_model
    return dict_model


def get_feature_meta(target_layer: str, feature_id: int) -> Dict:
    candidates = [
        target_layer,
        target_layer.replace(".", "_"),
        target_layer.replace("model.layers.", ""),
        target_layer.replace("transformer.", ""),
        target_layer.replace("transformer.h.", ""),
    ]
    layer_dict = {}
    for cand in candidates:
        if cand in state.feature_metadata:
            layer_dict = state.feature_metadata[cand]
            break
    return layer_dict.get(str(feature_id), layer_dict.get(feature_id, {}))


def get_top_activating_snippets(dict_model: torch.nn.Module, target_layer: str, feature_id: int) -> List[Dict]:
    """Computes or retrieves top activating context snippets with token-level activation values."""
    meta = get_feature_meta(target_layer, feature_id)
    if "snippets" in meta and meta["snippets"]:
        return meta["snippets"]

    if state.model is None or state.tokenizer is None:
        return []

    snippets = []
    device = state.model.device if hasattr(state.model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    hook_mgr = HookManager(state.model)
    hook_mgr.register_forward_hooks([target_layer])

    with torch.no_grad():
        for text in state.sample_contexts:
            inputs = state.tokenizer(text, return_tensors="pt").to(device)
            _ = state.model(**inputs)
            act = hook_mgr.activations[target_layer][0]
            scale = (dict_model.d_in ** 0.5) / (act.norm(dim=-1, keepdim=True) + 1e-8)
            normed_act = (act * scale).to(device=dict_model.get_decoder_weights().device, dtype=dict_model.get_decoder_weights().dtype)
            dict_model.eval()
            f = dict_model.encode(normed_act)  # (seq_len, d_sae)

            feat_acts = f[:, feature_id].tolist()
            max_act = max(feat_acts) if feat_acts else 0.0

            tokens = [
                {"token": state.tokenizer.decode([tid]), "act": float(a)}
                for tid, a in zip(inputs["input_ids"][0], feat_acts)
            ]

            snippets.append({
                "context_text": text,
                "max_activation": float(max_act),
                "tokens": tokens
            })

    hook_mgr.remove_hooks()
    snippets.sort(key=lambda s: s["max_activation"], reverse=True)
    top_snippets = snippets[:4]
    meta["snippets"] = top_snippets  # Cache for future instant lookups
    return top_snippets


async def health_handler(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "online",
        "model_loaded": state.model is not None,
        "active_layer": state.active_hook_point,
        "available_layers": list(state.available_layers.keys()),
        "total_layers": len(state.available_layers),
    })


async def layers_handler(request: web.Request) -> web.Response:
    return web.json_response({
        "active_layer": state.active_hook_point,
        "layers": list(state.available_layers.keys()),
    })


async def select_layer_handler(request: web.Request) -> web.Response:
    data = await request.json()
    hook_point = data.get("hook_point") or data.get("layer")
    dict_model = get_or_load_dictionary(hook_point)
    if dict_model is None:
        return web.json_response({"error": f"Layer '{hook_point}' not found in checkpoints."}, status=404)

    state.active_hook_point = hook_point
    return web.json_response({"status": "success", "active_layer": state.active_hook_point})


async def features_handler(request: web.Request) -> web.Response:
    page = int(request.query.get("page", 1))
    page_size = int(request.query.get("page_size", 40))
    search = request.query.get("search", None)
    target_layer = request.query.get("layer") or state.active_hook_point

    dict_model = get_or_load_dictionary(target_layer)
    if dict_model is None:
        return web.json_response({"error": f"Layer '{target_layer}' is not available."}, status=400)

    total_features = dict_model.d_sae

    if search:
        search_lower = search.lower()
        matching_indices = [
            idx for idx in range(total_features)
            if search_lower in get_feature_meta(target_layer, idx).get("explanation", f"Feature #{idx}").lower() or str(idx) == search_lower
        ]
        total_features = len(matching_indices)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_features)
        page_indices = matching_indices[start_idx:end_idx]
    else:
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_features)
        page_indices = list(range(start_idx, end_idx))

    features = []
    for idx in page_indices:
        fmeta = get_feature_meta(target_layer, idx)
        features.append({
            "feature_id": idx,
            "title": fmeta.get("title", f"Feature #{idx}"),
            "description": fmeta.get("description", ""),
            "explanation": fmeta.get("explanation", f"Feature #{idx}"),
            "l0_firing_rate": fmeta.get("firing_rate", 0.0),
            "max_activation": fmeta.get("max_activation", 0.0),
            "top_promoted_tokens": fmeta.get("top_promoted_tokens", []),
        })

    return web.json_response({
        "layer": target_layer,
        "total_features": total_features,
        "page": page,
        "page_size": page_size,
        "features": features,
    })


async def feature_details_handler(request: web.Request) -> web.Response:
    feature_id = int(request.match_info["feature_id"])
    top_k = int(request.query.get("top_k", 10))
    target_layer = request.query.get("layer") or state.active_hook_point

    meta = get_feature_meta(target_layer, feature_id)
    promoted = meta.get("top_promoted_tokens")
    suppressed = meta.get("top_suppressed_tokens")

    dict_model = get_or_load_dictionary(target_layer)
    if dict_model is None:
        return web.json_response({"error": f"Layer '{target_layer}' not loaded."}, status=400)

    # Fast path: Use precomputed attribution if available
    if not promoted or not suppressed:
        w_dec = dict_model.get_decoder_weights()
        if feature_id >= w_dec.shape[0]:
            return web.json_response({"error": "Feature ID out of bounds."}, status=404)

        promoted, suppressed = compute_direct_logit_attribution(
            decoder_vector=w_dec[feature_id],
            unembedding_weights=state.unembedding_weights,
            tokenizer=state.tokenizer,
            top_k=top_k,
        )
        meta["top_promoted_tokens"] = promoted
        meta["top_suppressed_tokens"] = suppressed

    snippets = get_top_activating_snippets(dict_model, target_layer, feature_id)
    max_act = meta.get("max_activation", max([s["max_activation"] for s in snippets] + [0.0]))
    firing_rate = meta.get("firing_rate", 0.001)

    return web.json_response({
        "feature_id": feature_id,
        "layer": target_layer,
        "title": meta.get("title", f"Feature #{feature_id}"),
        "description": meta.get("description", ""),
        "explanation": meta.get("explanation", f"Feature #{feature_id}"),
        "max_activation": float(max_act),
        "firing_rate": float(firing_rate),
        "snippets": snippets,
        "promoted_tokens": promoted,
        "suppressed_tokens": suppressed,
    })


async def steer_handler(request: web.Request) -> web.Response:
    if state.model is None or state.tokenizer is None:
        return web.json_response({"error": "Base model not loaded."}, status=400)

    data = await request.json()
    prompt = data.get("prompt", "")
    steered_features = {int(k): float(v) for k, v in data.get("steered_features", {}).items()}
    max_new_tokens = int(data.get("max_new_tokens", 40))
    temperature = float(data.get("temperature", 0.7))
    target_layer = data.get("layer") or state.active_hook_point

    dict_model = get_or_load_dictionary(target_layer)
    if dict_model is None:
        return web.json_response({"error": f"Layer '{target_layer}' not loaded for steering."}, status=400)

    steering_engine = FeatureSteeringEngine(
        model=state.model,
        tokenizer=state.tokenizer,
        dictionary_model=dict_model,
        hook_point=target_layer,
    )

    generated = steering_engine.generate_with_steering(
        prompt=prompt,
        steered_features=steered_features,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return web.json_response({
        "prompt": prompt,
        "generated_text": generated,
        "steered_features": steered_features,
        "layer": target_layer
    })


async def analyze_text_handler(request: web.Request) -> web.Response:
    """
    Neuronpedia Interactive Prompt Analyzer:
    Computes exact per-token activations and top-firing features for each token position.
    """
    if state.model is None or state.tokenizer is None:
        return web.json_response({"error": "Base model not loaded."}, status=400)

    data = await request.json()
    text = data.get("text", "")
    top_k_per_token = int(data.get("top_k_per_token", 12))
    target_layer = data.get("layer") or state.active_hook_point

    dict_model = get_or_load_dictionary(target_layer)
    if dict_model is None:
        return web.json_response({"error": f"Layer '{target_layer}' is not available."}, status=400)

    device = state.model.device if hasattr(state.model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    inputs = state.tokenizer(text, return_tensors="pt").to(device)
    hook_mgr = HookManager(state.model)
    hook_mgr.register_forward_hooks([target_layer])

    with torch.no_grad():
        _ = state.model(**inputs)
        act = hook_mgr.activations[target_layer][0]
        # Match training-time activation normalization (scale = sqrt(d_in) / norm)
        scale = (dict_model.d_in ** 0.5) / (act.norm(dim=-1, keepdim=True) + 1e-8)
        normed_act = (act * scale).to(device=dict_model.get_decoder_weights().device, dtype=dict_model.get_decoder_weights().dtype)
        dict_model.eval()
        f = dict_model.encode(normed_act)  # (seq_len, d_sae)

    hook_mgr.remove_hooks()
    token_ids = inputs["input_ids"][0].tolist()
    seq_len = len(token_ids)

    token_analysis = []
    for pos in range(seq_len):
        t_id = token_ids[pos]
        t_str = state.tokenizer.decode([t_id])
        pos_acts = f[pos]  # (d_sae,)

        # Non-zero active features on this specific token
        active_mask = pos_acts > 1e-4
        total_active = int(active_mask.sum().item())

        top_vals, top_indices = torch.topk(pos_acts, k=min(top_k_per_token, pos_acts.shape[-1]))
        
        token_top_features = []
        for val, feat_idx in zip(top_vals.tolist(), top_indices.tolist()):
            if val <= 1e-4:
                continue
            fmeta = get_feature_meta(target_layer, feat_idx)
            title = fmeta.get("title", f"Feature #{feat_idx}")
            exp = fmeta.get("explanation", f"Feature #{feat_idx}")
            promoted = fmeta.get("top_promoted_tokens", [])
            token_top_features.append({
                "feature_id": int(feat_idx),
                "title": title,
                "description": fmeta.get("description", ""),
                "activation": float(val),
                "explanation": exp,
                "top_promoted_tokens": promoted,
            })

        token_analysis.append({
            "token_index": pos,
            "token_str": t_str,
            "token_id": t_id,
            "total_active_features": total_active,
            "max_activation": float(top_vals[0].item()) if len(top_vals) > 0 else 0.0,
            "top_features": token_top_features,
        })

    return web.json_response({
        "text": text,
        "layer": target_layer,
        "total_tokens": seq_len,
        "tokens": token_analysis,
    })


async def attribution_graph_handler(request: web.Request) -> web.Response:
    """Computes an Anthropic Attribution Graph for an input prompt and target token."""
    if state.model is None or state.tokenizer is None:
        return web.json_response({"error": "Base model is not loaded."}, status=400)

    try:
        data = await request.json()
    except Exception:
        data = {}

    prompt = data.get("prompt", "The capital of France is")
    target_token_id = data.get("target_token_id", None)
    if target_token_id is not None:
        try:
            target_token_id = int(target_token_id)
        except (ValueError, TypeError):
            target_token_id = None

    pruning_threshold = float(data.get("pruning_threshold", 0.80))
    max_nodes = int(data.get("max_nodes", 40))
    max_edges = int(data.get("max_edges", 50))

    if state.attribution_engine is None:
        from src.core.multi_dictionary import MultiLayerDictionary
        from src.circuits.transcoder_circuit import AnthropicAttributionGraphEngine

        chosen_dir = None
        if state.checkpoint_dir and os.path.exists(os.path.join(state.checkpoint_dir, "multi_sae_config.json")):
            chosen_dir = state.checkpoint_dir
        else:
            trans_candidates = [
                "checkpoints/gemma3_270m_skip_transcoder_all_layers/step_25000",
                "checkpoints/planckgpt_spherical_tree_sasa/step_25000",
                state.checkpoint_dir if state.checkpoint_dir and "transcoder" in state.checkpoint_dir.lower() else None,
            ]
            for cand in trans_candidates:
                if cand and os.path.exists(cand):
                    chosen_dir = cand
                    break

        if not chosen_dir:
            return web.json_response({"error": "No multi-layer dictionary checkpoints found for attribution graphs."}, status=404)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = next(state.model.parameters()).dtype
        transcoders = MultiLayerDictionary.from_pretrained(chosen_dir, device=device, dtype=dtype)
        state.attribution_engine = AnthropicAttributionGraphEngine(
            model=state.model,
            tokenizer=state.tokenizer,
            transcoders=transcoders,
            feature_metadata=state.feature_metadata,
        )

    try:
        graph = state.attribution_engine.trace_graph(
            prompt=prompt,
            target_token_id=target_token_id,
            pruning_threshold=pruning_threshold,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        return web.json_response(graph)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/api/health", health_handler)
    app.router.add_get("/api/layers", layers_handler)
    app.router.add_post("/api/select_layer", select_layer_handler)
    app.router.add_get("/api/features", features_handler)
    app.router.add_get("/api/feature/{feature_id}", feature_details_handler)
    app.router.add_post("/api/steer", steer_handler)
    app.router.add_post("/api/analyze", analyze_text_handler)
    app.router.add_post("/api/attribution_graph", attribution_graph_handler)

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        async def index_handler(request: web.Request) -> web.FileResponse:
            return web.FileResponse(os.path.join(static_dir, "index.html"))
        app.router.add_get("/", index_handler)
        app.router.add_static("/", path=static_dir, name="static")

    return app
