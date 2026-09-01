import json
import os
from typing import Dict, Optional
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

    @property
    def dictionary_model(self):
        if self.active_hook_point in self.loaded_dictionaries:
            return self.loaded_dictionaries[self.active_hook_point]
        return None


state = DashboardState()


async def health_handler(request):
    return web.json_response({
        "status": "online",
        "model_loaded": state.model is not None,
        "active_layer": state.active_hook_point,
        "available_layers": list(state.available_layers.keys()),
        "total_layers": len(state.available_layers),
    })


async def layers_handler(request):
    return web.json_response({
        "active_layer": state.active_hook_point,
        "layers": list(state.available_layers.keys()),
    })


async def select_layer_handler(request):
    data = await request.json()
    hook_point = data.get("hook_point")
    if hook_point not in state.available_layers:
        return web.json_response({"error": f"Layer '{hook_point}' not found in available checkpoints."}, status=404)

    # Lazy-load layer dictionary into cache
    if hook_point not in state.loaded_dictionaries:
        subfolder = state.available_layers[hook_point]
        cfg_path = os.path.join(subfolder, "config.json")
        if not os.path.exists(cfg_path):
            return web.json_response({"error": f"config.json missing in {subfolder}"}, status=500)
        
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cls = get_dictionary_cls(cfg.get("class_name", "TopKSAE"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state.loaded_dictionaries[hook_point] = cls.from_pretrained(subfolder, device=device)

    state.active_hook_point = hook_point
    return web.json_response({"status": "success", "active_layer": state.active_hook_point})


async def features_handler(request):
    page = int(request.query.get("page", 1))
    page_size = int(request.query.get("page_size", 50))
    search = request.query.get("search", None)
    target_layer = request.query.get("layer") or state.active_hook_point

    if not target_layer or target_layer not in state.available_layers:
        return web.json_response({"error": "No active layer selected."}, status=400)

    # Ensure dictionary is loaded
    if target_layer not in state.loaded_dictionaries:
        subfolder = state.available_layers[target_layer]
        cfg_path = os.path.join(subfolder, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cls = get_dictionary_cls(cfg.get("class_name", "TopKSAE"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state.loaded_dictionaries[target_layer] = cls.from_pretrained(subfolder, device=device)

    dict_model = state.loaded_dictionaries[target_layer]
    total_features = dict_model.d_sae
    layer_meta = state.feature_metadata.get(target_layer, {})

    if search:
        search_lower = search.lower()
        matching_indices = []
        for idx in range(total_features):
            meta = layer_meta.get(idx, {})
            exp = meta.get("explanation", f"Feature #{idx}")
            if search_lower in exp.lower() or str(idx) == search_lower:
                matching_indices.append(idx)
        
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
        meta = layer_meta.get(idx, {})
        features.append({
            "feature_id": idx,
            "explanation": meta.get("explanation", f"Feature #{idx}"),
            "l0_firing_rate": meta.get("firing_rate", 0.0),
            "max_activation": meta.get("max_activation", 0.0),
            "top_promoted_tokens": meta.get("top_promoted_tokens", []),
        })

    return web.json_response({
        "layer": target_layer,
        "total_features": total_features,
        "page": page,
        "page_size": page_size,
        "features": features,
    })


async def feature_logits_handler(request):
    feature_id = int(request.match_info["feature_id"])
    top_k = int(request.query.get("top_k", 10))
    target_layer = request.query.get("layer") or state.active_hook_point

    if not target_layer or target_layer not in state.loaded_dictionaries:
        if target_layer in state.available_layers:
            subfolder = state.available_layers[target_layer]
            with open(os.path.join(subfolder, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cls = get_dictionary_cls(cfg.get("class_name", "TopKSAE"))
            state.loaded_dictionaries[target_layer] = cls.from_pretrained(subfolder, device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            return web.json_response({"error": "Dictionary model not loaded."}, status=400)

    dict_model = state.loaded_dictionaries[target_layer]
    w_dec = dict_model.get_decoder_weights()
    if feature_id >= w_dec.shape[0]:
        return web.json_response({"error": "Feature ID out of bounds."}, status=404)

    d_v = w_dec[feature_id]
    promoted, suppressed = compute_direct_logit_attribution(
        decoder_vector=d_v,
        unembedding_weights=state.unembedding_weights,
        tokenizer=state.tokenizer,
        top_k=top_k,
    )
    meta = state.feature_metadata.get(target_layer, {}).get(feature_id, {})
    return web.json_response({
        "feature_id": feature_id,
        "layer": target_layer,
        "explanation": meta.get("explanation", f"Feature #{feature_id}"),
        "promoted_tokens": promoted,
        "suppressed_tokens": suppressed,
    })


async def steer_handler(request):
    if state.model is None or state.tokenizer is None:
        return web.json_response({"error": "Base model not loaded."}, status=400)

    data = await request.json()
    prompt = data.get("prompt", "")
    steered_features = {int(k): float(v) for k, v in data.get("steered_features", {}).items()}
    max_new_tokens = int(data.get("max_new_tokens", 40))
    temperature = float(data.get("temperature", 0.7))
    target_layer = data.get("layer") or state.active_hook_point

    if target_layer not in state.loaded_dictionaries:
        if target_layer in state.available_layers:
            subfolder = state.available_layers[target_layer]
            with open(os.path.join(subfolder, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cls = get_dictionary_cls(cfg.get("class_name", "TopKSAE"))
            state.loaded_dictionaries[target_layer] = cls.from_pretrained(subfolder, device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            return web.json_response({"error": "Dictionary not loaded for steering."}, status=400)

    dict_model = state.loaded_dictionaries[target_layer]
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
    return web.json_response({"prompt": prompt, "generated_text": generated, "steered_features": steered_features, "layer": target_layer})


async def analyze_text_handler(request):
    if state.model is None or state.tokenizer is None:
        return web.json_response({"error": "Base model not loaded."}, status=400)

    data = await request.json()
    text = data.get("text", "")
    top_k_features = int(data.get("top_k_features", 8))
    target_layer = data.get("layer") or state.active_hook_point

    if target_layer not in state.loaded_dictionaries:
        if target_layer in state.available_layers:
            subfolder = state.available_layers[target_layer]
            with open(os.path.join(subfolder, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cls = get_dictionary_cls(cfg.get("class_name", "TopKSAE"))
            state.loaded_dictionaries[target_layer] = cls.from_pretrained(subfolder, device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            return web.json_response({"error": "Dictionary not loaded."}, status=400)

    dict_model = state.loaded_dictionaries[target_layer]
    inputs = state.tokenizer(text, return_tensors="pt").to(state.model.device)
    hook_mgr = HookManager(state.model)
    hook_mgr.register_forward_hooks([target_layer])

    with torch.no_grad():
        _ = state.model(**inputs)
        act = hook_mgr.activations[target_layer][0]
        f = dict_model.encode(act)

    hook_mgr.remove_hooks()
    tokens = [state.tokenizer.decode([tid]) for tid in inputs["input_ids"][0]]

    mean_acts = f.mean(dim=0)
    top_vals, top_indices = torch.topk(mean_acts, k=min(top_k_features, f.shape[-1]))

    token_heatmaps = []
    layer_meta = state.feature_metadata.get(target_layer, {})
    for feat_idx in top_indices.tolist():
        feat_acts = f[:, feat_idx].tolist()
        exp = layer_meta.get(feat_idx, {}).get("explanation", f"Feature #{feat_idx}")
        token_heatmaps.append({
            "feature_id": feat_idx,
            "explanation": exp,
            "mean_activation": float(mean_acts[feat_idx].item()),
            "token_activations": [{"token": t, "act": a} for t, a in zip(tokens, feat_acts)],
        })

    return web.json_response({"text": text, "layer": target_layer, "top_features": token_heatmaps})


def create_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/api/health", health_handler)
    app.router.add_get("/api/layers", layers_handler)
    app.router.add_post("/api/select_layer", select_layer_handler)
    app.router.add_get("/api/features", features_handler)
    app.router.add_get("/api/feature/{feature_id}/logits", feature_logits_handler)
    app.router.add_post("/api/steer", steer_handler)
    app.router.add_post("/api/analyze_text", analyze_text_handler)

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        # Serve index.html at root
        async def index_handler(request):
            return web.FileResponse(os.path.join(static_dir, "index.html"))
        app.router.add_get("/", index_handler)
        app.router.add_static("/", path=static_dir, name="static")

    return app
