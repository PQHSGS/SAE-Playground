import os
from typing import Dict, Optional
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.utils.hf_helpers import compute_direct_logit_attribution
from src.circuits.steering import FeatureSteeringEngine
from src.core.hook_manager import HookManager

app = FastAPI(title="Neuronpedia Interpretability Playground API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
class DashboardState:
    model = None
    tokenizer = None
    dictionary_model = None
    hook_point = "transformer.h.6"
    unembedding_weights = None
    feature_metadata = {}

state = DashboardState()


class SteeringRequest(BaseModel):
    prompt: str
    steered_features: Dict[int, float]  # {feat_id: alpha}
    max_new_tokens: int = 40
    temperature: float = 0.7


class TextAnalysisRequest(BaseModel):
    text: str
    top_k_features: int = 5


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": state.model is not None,
        "dictionary_loaded": state.dictionary_model is not None,
        "hook_point": state.hook_point,
    }


@app.get("/api/features")
def get_features(page: int = 1, page_size: int = 50, search: Optional[str] = None):
    if state.dictionary_model is None:
        raise HTTPException(status_code=400, detail="Dictionary model not loaded.")

    total_features = state.dictionary_model.d_sae
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_features)

    features = []
    for idx in range(start_idx, end_idx):
        meta = state.feature_metadata.get(idx, {})
        features.append({
            "feature_id": idx,
            "explanation": meta.get("explanation", f"Feature #{idx}"),
            "l0_firing_rate": meta.get("firing_rate", 0.0),
            "max_activation": meta.get("max_activation", 0.0),
        })

    return {
        "total_features": total_features,
        "page": page,
        "page_size": page_size,
        "features": features,
    }


@app.get("/api/feature/{feature_id}/logits")
def get_feature_logits(feature_id: int, top_k: int = 10):
    if state.dictionary_model is None or state.unembedding_weights is None or state.tokenizer is None:
        raise HTTPException(status_code=400, detail="Model and dictionary must be initialized.")

    w_dec = state.dictionary_model.get_decoder_weights()
    if feature_id >= w_dec.shape[0]:
        raise HTTPException(status_code=404, detail="Feature ID out of bounds.")

    d_v = w_dec[feature_id]
    promoted, suppressed = compute_direct_logit_attribution(
        decoder_vector=d_v,
        unembedding_weights=state.unembedding_weights,
        tokenizer=state.tokenizer,
        top_k=top_k,
    )
    return {"feature_id": feature_id, "promoted_tokens": promoted, "suppressed_tokens": suppressed}


@app.post("/api/steer")
def steer_generation(req: SteeringRequest):
    if state.model is None or state.tokenizer is None or state.dictionary_model is None:
        raise HTTPException(status_code=400, detail="Model and dictionary must be loaded for steering.")

    steering_engine = FeatureSteeringEngine(
        model=state.model,
        tokenizer=state.tokenizer,
        dictionary_model=state.dictionary_model,
        hook_point=state.hook_point,
    )

    generated = steering_engine.generate_with_steering(
        prompt=req.prompt,
        steered_features=req.steered_features,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
    )
    return {"prompt": req.prompt, "generated_text": generated, "steered_features": req.steered_features}


@app.post("/api/analyze_text")
def analyze_text(req: TextAnalysisRequest):
    if state.model is None or state.tokenizer is None or state.dictionary_model is None:
        raise HTTPException(status_code=400, detail="Model and dictionary must be loaded.")

    inputs = state.tokenizer(req.text, return_tensors="pt").to(state.model.device)
    hook_mgr = HookManager(state.model)
    hook_mgr.register_forward_hooks([state.hook_point])

    with torch.no_grad():
        _ = state.model(**inputs)
        act = hook_mgr.activations[state.hook_point][0]  # (seq_len, d_in)
        f = state.dictionary_model.encode(act)           # (seq_len, d_sae)

    hook_mgr.remove_hooks()

    tokens = [state.tokenizer.decode([tid]) for tid in inputs["input_ids"][0]]

    # Top features firing in this text
    mean_acts = f.mean(dim=0)
    top_vals, top_indices = torch.topk(mean_acts, k=min(req.top_k_features, f.shape[-1]))

    token_heatmaps = []
    for feat_idx in top_indices.tolist():
        feat_acts = f[:, feat_idx].tolist()
        token_heatmaps.append({
            "feature_id": feat_idx,
            "explanation": state.feature_metadata.get(feat_idx, {}).get("explanation", f"Feature #{feat_idx}"),
            "tokens": [{"token": tok, "activation": act} for tok, act in zip(tokens, feat_acts)]
        })

    return {"tokens": tokens, "top_features": token_heatmaps}


# Mount static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
