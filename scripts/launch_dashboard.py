import argparse
import os
import sys
import yaml
import uvicorn
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dashboard.server import app, state
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
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model_name_or_path
    hook_point = args.hook_point
    checkpoint_dir = args.checkpoint_dir

    if args.config:
        cfg = load_yaml(args.config)
        model_name = model_name or cfg.get("model_name_or_path", "gpt2")
        hook_point = hook_point or cfg.get("hook_points", ["transformer.h.6"])[0]
        if not checkpoint_dir:
            out_dir = cfg.get("output_dir", "checkpoints")
            if os.path.exists(out_dir):
                subdirs = [os.path.join(out_dir, d) for d in os.listdir(out_dir) if d.startswith("step_")]
                if subdirs:
                    subdirs.sort(key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
                    checkpoint_dir = subdirs[-1]
                else:
                    checkpoint_dir = out_dir

    model_name = model_name or "gpt2"
    hook_point = hook_point or "transformer.h.6"

    print(f"Initializing Playground Dashboard on http://localhost:{args.port}...")
    
    # Load LLM
    print(f"Loading base LLM '{model_name}'...")
    model, tokenizer = load_model_and_tokenizer(model_name, device_map="auto")
    state.model = model
    state.tokenizer = tokenizer
    state.hook_point = hook_point
    state.unembedding_weights = get_unembedding_weights(model)

    # Load Dictionary if provided
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        import json
        with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
            cfg = json.load(f)
        cls_name = cfg.get("class_name", "TopKSAE")
        cls = get_dictionary_cls(cls_name)
        state.dictionary_model = cls.from_pretrained(checkpoint_dir, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loaded {cls_name} dictionary from {checkpoint_dir} (d_sae={state.dictionary_model.d_sae})")
    else:
        # Build lightweight demo TopK dictionary
        from src.architectures.sae.topk_sae import TopKSAE
        d_in = state.unembedding_weights.shape[0] if state.unembedding_weights.ndim == 2 else 768
        state.dictionary_model = TopKSAE(d_in=d_in, d_sae=d_in * 4, k=16).to(model.device)
        print(f"Created demo TopK dictionary: d_in={d_in}, d_sae={d_in * 4}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
