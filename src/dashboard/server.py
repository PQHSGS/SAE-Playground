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


def get_top_activating_snippets(dict_model: torch.nn.Module, target_layer: str, feature_id: int) -> List[Dict]:
    """Computes or retrieves top activating context snippets with token-level activation values."""
    if state.model is None or state.tokenizer is None:
        return []

    meta = state.feature_metadata.get(target_layer, {}).get(feature_id, {})
    if "snippets" in meta and meta["snippets"]:
        return meta["snippets"]

    snippets = []
    device = state.model.device if hasattr(state.model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    hook_mgr = HookManager(state.model)
    hook_mgr.register_forward_hooks([target_layer])

    with torch.no_grad():
        for text in state.sample_contexts:
            inputs = state.tokenizer(text, return_tensors="pt").to(device)
            _ = state.model(**inputs)
            act = hook_mgr.activations[target_layer][0]
            f = dict_model.encode(act)  # (seq_len, d_sae)

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
    # Sort snippets by max activation descending
    snippets.sort(key=lambda s: s["max_activation"], reverse=True)
    return snippets[:4]


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
    hook_point = data.get("hook_point")
    dict_model = get_or_load_dictionary(hook_point)
    if dict_model is None:
        return web.json_response({"error": f"Layer '{hook_point}' not found in checkpoints."}, status=404)

    state.active_hook_point = hook_point
    return web.json_response({"status": "success", "active_layer": state.active_hook_point})


async def features_handler(request: web.Request) -> web.Response:
    page = int(request.query.get("page", 1))
    page_size = int(request.query.get("page_size", 50))
    search = request.query.get("search", None)
    target_layer = request.query.get("layer") or state.active_hook_point

    dict_model = get_or_load_dictionary(target_layer)
    if dict_model is None:
        return web.json_response({"error": f"Layer '{target_layer}' is not available."}, status=400)

    total_features = dict_model.d_sae
    layer_meta = state.feature_metadata.get(target_layer, {})

    if search:
        search_lower = search.lower()
        matching_indices = [
            idx for idx in range(total_features)
            if search_lower in layer_meta.get(idx, {}).get("explanation", f"Feature #{idx}").lower() or str(idx) == search_lower
        ]
        total_features = len(matching_indices)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_features)
        page_indices = matching_indices[start_idx:end_idx]
    else:
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_features)
        page_indices = list(range(start_idx, end_idx))

    features = [
        {
            "feature_id": idx,
            "explanation": layer_meta.get(idx, {}).get("explanation", f"Feature #{idx}"),
            "l0_firing_rate": layer_meta.get(idx, {}).get("firing_rate", 0.0),
            "max_activation": layer_meta.get(idx, {}).get("max_activation", 0.0),
            "top_promoted_tokens": layer_meta.get(idx, {}).get("top_promoted_tokens", []),
        }
        for idx in page_indices
    ]

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

    dict_model = get_or_load_dictionary(target_layer)
    if dict_model is None:
        return web.json_response({"error": f"Layer '{target_layer}' not loaded."}, status=400)

    w_dec = dict_model.get_decoder_weights()
    if feature_id >= w_dec.shape[0]:
        return web.json_response({"error": "Feature ID out of bounds."}, status=404)

    promoted, suppressed = compute_direct_logit_attribution(
        decoder_vector=w_dec[feature_id],
        unembedding_weights=state.unembedding_weights,
        tokenizer=state.tokenizer,
        top_k=top_k,
    )
    meta = state.feature_metadata.get(target_layer, {}).get(feature_id, {})
    snippets = get_top_activating_snippets(dict_model, target_layer, feature_id)

    max_act = meta.get("max_activation", max([s["max_activation"] for s in snippets] + [0.0]))
    firing_rate = meta.get("firing_rate", 0.001)

    return web.json_response({
        "feature_id": feature_id,
        "layer": target_layer,
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
    if state.model is None or state.tokenizer is None:
        return web.json_response({"error": "Base model not loaded."}, status=400)

    data = await request.json()
    text = data.get("text", "")
    top_k_features = int(data.get("top_k_features", 8))
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
        f = dict_model.encode(act)

    hook_mgr.remove_hooks()
    tokens = [state.tokenizer.decode([tid]) for tid in inputs["input_ids"][0]]

    mean_acts = f.mean(dim=0)
    _, top_indices = torch.topk(mean_acts, k=min(top_k_features, f.shape[-1]))

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
    app.router.add_get("/api/feature/{feature_id}", feature_details_handler)
    app.router.add_get("/api/feature/{feature_id}/logits", feature_details_handler)
    app.router.add_post("/api/steer", steer_handler)
    app.router.add_post("/api/analyze", analyze_text_handler)

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        async def index_handler(request: web.Request) -> web.FileResponse:
            return web.FileResponse(os.path.join(static_dir, "index.html"))
        app.router.add_get("/", index_handler)
        app.router.add_static("/", path=static_dir, name="static")

    return app
