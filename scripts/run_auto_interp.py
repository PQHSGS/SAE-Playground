import argparse
import os
import sys
import yaml
import torch
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.hf_helpers import load_model_and_tokenizer
from src.architectures.registry import get_dictionary_cls
from src.core.activation_buffer import ActivationBuffer
from src.auto_interpret.sample_collector import FeatureSampleCollector
from src.auto_interpret.explainer import FeatureExplainer
from src.utils.io import save_json

load_dotenv()


def load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Run automated interpretation & explanation scoring via YAML config or CLI flags.")
    parser.add_argument("--config", type=str, default=None, help="Path to experiment YAML config file")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--hook_point", type=str, default=None)
    parser.add_argument("--num_features", type=int, default=20, help="Number of features to explain")
    parser.add_argument("--output_file", type=str, default="data/auto_interp_results.json")
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

    import json
    with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    cls = get_dictionary_cls(cfg.get("class_name", "TopKSAE"))
    dict_model = cls.from_pretrained(checkpoint_dir, device=device)

    model, tokenizer = load_model_and_tokenizer(model_name, device_map="auto")

    collector = FeatureSampleCollector(dictionary_model=dict_model, tokenizer=tokenizer)
    explainer = FeatureExplainer(provider="heuristic")

    print(f"Collecting max-activating examples for first {args.num_features} features...")
    buffer = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        hook_points=[hook_point],
        batch_size=1024,
        buffer_size=4096,
        device=device,
    )

    for _ in range(5):
        sample_batch = buffer.next_batch()
        if isinstance(sample_batch, (tuple, list)):
            sample_batch = sample_batch[0]
        # Fake sequential input IDs for collector
        dummy_ids = torch.arange(sample_batch.shape[0]) % tokenizer.vocab_size
        collector.process_sequence_activations(
            input_ids=dummy_ids,
            activations=sample_batch,
            target_features=list(range(args.num_features))
        )

    results = {}
    for feat_id in range(args.num_features):
        snippets = collector.get_feature_snippets(feat_id)
        explanation = explainer.explain_feature(feat_id, snippets)
        results[feat_id] = {
            "feature_id": feat_id,
            "explanation": explanation,
            "snippet_count": len(snippets),
        }
        print(f"Feature #{feat_id:3d}: {explanation}")

    save_json(results, args.output_file)
    print(f"\nSaved explanation results to {args.output_file}")


if __name__ == "__main__":
    main()
