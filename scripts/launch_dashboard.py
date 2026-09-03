import argparse
import json
import os
import sys
import yaml
from aiohttp import web

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dashboard.server import create_app, state
from src.architectures.registry import get_dictionary_cls
from src.utils.hf_helpers import load_model_and_tokenizer, get_unembedding_weights


def load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the local Neuronpedia Interactive UI via YAML config or CLI flags.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML config file")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="HuggingFace model ID")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to dictionary checkpoint folder")
    parser.add_argument("--hook_point", type=str, default=None, help="Submodule hook point")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--device", type=str, default="cpu", help="Device for dashboard (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model_name_or_path
    hook_point = args.hook_point
    checkpoint_dir = args.checkpoint_dir

    if args.config:
        cfg = load_yaml(args.config)
        model_name = model_name or cfg.get("model_name_or_path", "google/gemma-3-270m")
        hook_point = hook_point or cfg.get("hook_points", ["model.layers.9"])[0]
        if not checkpoint_dir:
            out_dir = cfg.get("output_dir", "checkpoints/gemma3_270m_resid")
            if os.path.exists(out_dir):
                subdirs = [os.path.join(out_dir, d) for d in os.listdir(out_dir) if d.startswith("step_")]
                if subdirs:
                    subdirs.sort(key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
                    checkpoint_dir = subdirs[-1]
                else:
                    checkpoint_dir = out_dir

    model_name = model_name or "google/gemma-3-270m"
    hook_point = hook_point or "model.layers.9"
    checkpoint_dir = checkpoint_dir or "checkpoints/gemma3_270m_resid/step_25000"

    print(f"🚀 Initializing Playground Dashboard on http://localhost:{args.port} (device={args.device})...")
    
    # 1. Load Base LLM
    print(f"📦 Loading base LLM '{model_name}' on {args.device}...")
    model, tokenizer = load_model_and_tokenizer(model_name, device_map=args.device)
    state.model = model
    state.tokenizer = tokenizer
    state.unembedding_weights = get_unembedding_weights(model).to(args.device)

    # 2. Discover Available Layers (Multi-Layer or Single-Layer Checkpoint)
    available_layers = {}
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        multi_cfg_file = os.path.join(checkpoint_dir, "multi_sae_config.json")
        if os.path.exists(multi_cfg_file):
            with open(multi_cfg_file, "r", encoding="utf-8") as f:
                multi_cfg = json.load(f)
            layer_hooks = multi_cfg.get("hook_points", [])
            for hp in layer_hooks:
                safe_name = hp.replace(".", "_")
                subpath = os.path.join(checkpoint_dir, safe_name)
                if os.path.exists(subpath):
                    available_layers[hp] = subpath
        elif os.path.exists(os.path.join(checkpoint_dir, "config.json")):
            available_layers[hook_point] = checkpoint_dir

    state.available_layers = available_layers
    state.checkpoint_dir = checkpoint_dir

    model_slug = model_name.split("/")[-1].lower()
    base_slug = model_slug.split("-")[0].split("_")[0]
    raw_candidates = [
        f"data/{model_slug}_feature_metadata.json",
        f"data/{base_slug}_feature_metadata.json",
        f"data/{os.path.basename(checkpoint_dir)}_feature_metadata.json",
        "data/planckgpt_feature_metadata.json" if "planck" in model_slug else None,
        "data/gemma3_270m_feature_metadata.json" if "gemma" in model_slug else None,
        "data/feature_metadata.json",
        "data/auto_interp_results.json"
    ]
    metadata_candidates = [c for c in raw_candidates if c is not None]
    for meta_path in metadata_candidates:
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    loaded_meta = json.load(f)
                if isinstance(loaded_meta, dict):
                    print(f"Loaded feature metadata from {meta_path}")
                    is_multi_layer = any(isinstance(v, dict) and not str(k).isdigit() for k, v in loaded_meta.items())
                    if is_multi_layer:
                        # Nested multi-layer dictionary format { "model.layers.0": { "0": {...} } }
                        for lyr, meta_map in loaded_meta.items():
                            state.feature_metadata[lyr] = {str(k): v for k, v in meta_map.items()}
                    else:
                        # Flat single-layer format
                        state.feature_metadata[hook_point] = {str(k): v for k, v in loaded_meta.items()}
                break
            except Exception as e:
                print(f"Warning: Failed to load {meta_path}: {e}")

    # 4. Activate initial hook point
    if hook_point in state.available_layers:
        subfolder = state.available_layers[hook_point]
        with open(os.path.join(subfolder, "config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cls = get_dictionary_cls(cfg.get("class_name", "TopKSAE"))
        state.loaded_dictionaries[hook_point] = cls.from_pretrained(subfolder, device=args.device)
        state.active_hook_point = hook_point
        print(f"Active dictionary layer set to: '{hook_point}'")

    print(f"✨ Dashboard ready! Available Layers: {len(state.available_layers)}")
    app = create_app()
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
