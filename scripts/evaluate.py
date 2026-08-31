import argparse
import os
import sys
import json
import yaml
import torch
from rich.console import Console
from rich.table import Table

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.multi_dictionary import MultiLayerDictionary
from src.architectures.registry import get_dictionary_cls
from src.utils.hf_helpers import load_model_and_tokenizer
from src.core.activation_buffer import ActivationBuffer
from src.evaluation.metrics import compute_reconstruction_metrics
from src.evaluation.feature_stats import compute_feature_statistics


def load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark and evaluate trained dictionaries via YAML config or checkpoint dir.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML config file")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Direct directory containing checkpoints")
    parser.add_argument("--num_eval_tokens", type=int, default=16384)
    return parser.parse_args()


def main():
    args = parse_args()
    console = Console()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.config:
        cfg = load_yaml(args.config)
        output_dir = cfg.get("output_dir", "checkpoints")
        model_name = cfg.get("model_name_or_path", "gpt2")
        hook_points = cfg.get("hook_points", ["transformer.h.6"])

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
        hook_points = ["transformer.h.6"]
    else:
        raise ValueError("Please provide either --config or --checkpoint_dir")

    console.print(f"[bold green]Loading checkpoint from:[/] {checkpoint_dir}")
    is_multi = os.path.exists(os.path.join(checkpoint_dir, "multi_sae_config.json"))

    if is_multi:
        multi_dict = MultiLayerDictionary.from_pretrained(checkpoint_dir, device=device)
        hook_points = list(multi_dict.hook_point_map.values())
    else:
        with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
            dict_cfg = json.load(f)
        cls_name = dict_cfg.get("class_name", "TopKSAE")
        sae = get_dictionary_cls(cls_name).from_pretrained(checkpoint_dir, device=device)
        primary_hp = hook_points[0] if hook_points else "default"
        multi_dict = MultiLayerDictionary({primary_hp: sae})

    console.print(f"[bold blue]Loading base model:[/] {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name, device_map="auto")

    buffer = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        hook_points=hook_points,
        batch_size=args.num_eval_tokens,
        buffer_size=args.num_eval_tokens,
        device=device,
        return_dict=True,
    )

    batch_dict = buffer.next_batch()

    table = Table(title="Evaluation & Benchmark Summary across Layer Dictionaries")
    table.add_column("Layer / Hook Point", style="cyan", no_wrap=True)
    table.add_column("NMSE", justify="right", style="magenta")
    table.add_column("L0 (Active)", justify="right", style="green")
    table.add_column("Explained Var", justify="right", style="yellow")
    table.add_column("Dead Latents (%)", justify="right", style="red")

    for hp in hook_points:
        sae = multi_dict.get_dictionary(hp)
        acts = batch_dict[hp]
        recon_metrics = compute_reconstruction_metrics(sae, acts)
        feat_stats = compute_feature_statistics(sae, acts)

        table.add_row(
            hp,
            f"{recon_metrics['nmse']:.4f}",
            f"{recon_metrics['l0']:.1f}",
            f"{recon_metrics['explained_variance'] * 100:.1f}%",
            f"{feat_stats['dead_features_pct']:.1f}%",
        )

    console.print(table)


if __name__ == "__main__":
    main()
