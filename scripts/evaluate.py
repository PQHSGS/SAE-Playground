import argparse
import os
import sys
import json
import yaml
import torch
from rich.console import Console

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.multi_dictionary import MultiLayerDictionary
from src.architectures.registry import get_dictionary_cls
from src.utils.hf_helpers import load_model_and_tokenizer
from src.core.activation_buffer import ActivationBuffer
from src.evaluation.benchmark_suite import SAEBenchmarkSuite


def load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark and evaluate trained dictionaries via YAML config or checkpoint dir.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML config file")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Direct directory containing checkpoints")
    parser.add_argument("--num_eval_tokens", type=int, default=16384, help="Number of validation tokens to evaluate")
    parser.add_argument("--full_benchmark", action="store_true", default=False, help="Run complete multi-dimensional SAEBench suite (splitting, absorption, Gini, downstream faithfulness)")
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
    bench_suite = SAEBenchmarkSuite(console=console)

    test_samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Quantum mechanics reveals that particles exist in probabilistic wavefunctions.",
        "In artificial intelligence, transformer self-attention enables global contextual routing.",
        "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])",
    ]

    for hp in hook_points:
        sae = multi_dict.get_dictionary(hp)
        acts = batch_dict[hp]
        if isinstance(acts, (tuple, list)):
            x_in, targets = acts[0], acts[1]
        else:
            x_in, targets = acts, acts

        results = bench_suite.evaluate(
            dictionary_model=sae,
            activations=x_in,
            targets=targets,
            model=model if args.full_benchmark else None,
            tokenizer=tokenizer if args.full_benchmark else None,
            hook_point=hp if args.full_benchmark else None,
            test_texts=test_samples if args.full_benchmark else None,
        )

        bench_suite.print_report(results, title=f"SAE Benchmark Report: Layer '{hp}'")


if __name__ == "__main__":
    main()
