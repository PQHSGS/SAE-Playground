import argparse
import torch
from src.utils.hf_helpers import load_model_and_tokenizer
from src.architectures.registry import get_dictionary_cls
from src.circuits.steering import FeatureSteeringEngine


import os
import yaml


def load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Steering Demo CLI via YAML config or CLI flags")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML config file")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to dictionary checkpoint folder")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--hook_point", type=str, default=None)
    parser.add_argument("--feature_id", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=10.0, help="Steering coefficient (positive to boost, negative to suppress)")
    parser.add_argument("--prompt", type=str, default="The scientist walked into the laboratory and")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory '{checkpoint_dir}' does not exist. Specify --config or --checkpoint_dir.")

    model, tokenizer = load_model_and_tokenizer(model_name, device_map="auto")
    
    import json
    with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    cls = get_dictionary_cls(cfg.get("class_name", "TopKSAE"))
    dict_model = cls.from_pretrained(checkpoint_dir, device=device)

    engine = FeatureSteeringEngine(
        model=model,
        tokenizer=tokenizer,
        dictionary_model=dict_model,
        hook_point=hook_point,
    )

    print(f"\n--- Original Prompt ---\n{args.prompt}\n")
    
    # Unsteered baseline
    unsteered = engine.generate_with_steering(args.prompt, steered_features={}, max_new_tokens=40)
    print(f"--- Baseline Generation ---\n{unsteered}\n")

    # Steered generation
    steered = engine.generate_with_steering(args.prompt, steered_features={args.feature_id: args.alpha}, max_new_tokens=40)
    print(f"--- Steered Generation (Feature #{args.feature_id} with alpha={args.alpha:+.1f}) ---\n{steered}\n")


if __name__ == "__main__":
    main()
