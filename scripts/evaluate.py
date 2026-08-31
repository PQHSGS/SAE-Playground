import argparse
import os
import torch
from src.core.base_dictionary import BaseDictionary
from src.architectures.registry import get_dictionary_cls
from src.utils.hf_helpers import load_model_and_tokenizer
from src.core.activation_buffer import ActivationBuffer
from src.evaluation.metrics import compute_reconstruction_metrics, compute_ce_loss_recovery
from src.evaluation.feature_stats import compute_feature_statistics


import yaml


def load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark and evaluate trained dictionaries via YAML config or checkpoint dir.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML config file")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Direct directory containing config.json and model.safetensors")
    parser.add_argument("--num_eval_tokens", type=int, default=16384)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.config:
        cfg = load_yaml(args.config)
        output_dir = cfg.get("output_dir", "checkpoints")
        model_name = cfg.get("model_name_or_path", "gpt2")
        hook_point = cfg.get("hook_points", ["transformer.h.6"])[0]

        # Find latest step checkpoint in output_dir
        if os.path.exists(output_dir):
            subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("step_")]
            if subdirs:
                subdirs.sort(key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
                checkpoint_dir = subdirs[-1]
            else:
                checkpoint_dir = output_dir
        else:
            checkpoint_dir = output_dir
    elif args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
        model_name = "gpt2"
        hook_point = "transformer.h.6"
    else:
        raise ValueError("Please provide either --config (e.g. --config configs/gpt2_topk.yaml) or --checkpoint_dir")

    print(f"Loading dictionary from {checkpoint_dir}...")
    import json
    with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
        dict_cfg = json.load(f)

    cls_name = dict_cfg.get("class_name", "TopKSAE")
    cls = get_dictionary_cls(cls_name)
    dict_model = cls.from_pretrained(checkpoint_dir, device=device)

    print(f"Loading target LLM {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name, device_map="auto")

    buffer = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        hook_points=[hook_point],
        batch_size=args.num_eval_tokens,
        buffer_size=args.num_eval_tokens,
        device=device,
    )

    acts = buffer.next_batch()
    if isinstance(acts, (tuple, list)):
        acts = acts[0]

    print("\n--- 1. Reconstruction & Sparsity Metrics ---")
    recon_metrics = compute_reconstruction_metrics(dict_model, acts)
    for k, v in recon_metrics.items():
        print(f"  {k:20s}: {v:.6f}")

    print("\n--- 2. Feature Utilization & Dead Neurons ---")
    feat_stats = compute_feature_statistics(dict_model, acts)
    for k, v in feat_stats.items():
        print(f"  {k:20s}: {v}")

    print("\n--- 3. Downstream Cross-Entropy Loss Recovery ---")
    test_texts = [
        "In physics, spacetime is any mathematical model which fuses the three dimensions of space and the one dimension of time into a single four-dimensional continuum.",
        "The quick brown fox jumps over the lazy dog in the sunny meadow near the river bank.",
    ]
    ce_metrics = compute_ce_loss_recovery(
        model=model,
        tokenizer=tokenizer,
        dictionary_model=dict_model,
        hook_point=hook_point,
        test_texts=test_texts,
        device=device,
    )
    for k, v in ce_metrics.items():
        print(f"  {k:20s}: {v:.6f}")


if __name__ == "__main__":
    main()
