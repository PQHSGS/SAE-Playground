import argparse
import json
import os
import sys
import yaml
import torch
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.hf_helpers import load_model_and_tokenizer, get_unembedding_weights, compute_direct_logit_attribution
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
    parser.add_argument("--num_features", type=int, default=100, help="Number of features to explain")
    parser.add_argument("--output_file", type=str, default="data/feature_metadata.json")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

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

    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory '{checkpoint_dir}' does not exist.")

    # Multi-layer checkpoint handling
    layer_dir = checkpoint_dir
    multi_cfg_file = os.path.join(checkpoint_dir, "multi_sae_config.json")
    if os.path.exists(multi_cfg_file):
        safe_name = hook_point.replace(".", "_")
        layer_subpath = os.path.join(checkpoint_dir, safe_name)
        if os.path.exists(layer_subpath):
            layer_dir = layer_subpath

    cfg_file = os.path.join(layer_dir, "config.json")
    if not os.path.exists(cfg_file):
        raise ValueError(f"No config.json found in {layer_dir}")

    with open(cfg_file, "r") as f:
        cfg = json.load(f)
    cls_name = cfg.get("class_name", "BatchTopKSAE")
    cls = get_dictionary_cls(cls_name)
    dict_model = cls.from_pretrained(layer_dir, device=device)
    print(f"Loaded {cls_name} dictionary from {layer_dir} (d_sae={dict_model.d_sae})")

    print(f"Loading base LLM '{model_name}' on {device}...")
    model, tokenizer = load_model_and_tokenizer(model_name, device_map=device)
    unembed_weights = get_unembedding_weights(model).to(device)

    collector = FeatureSampleCollector(dictionary_model=dict_model, tokenizer=tokenizer)
    explainer = FeatureExplainer(provider="heuristic")

    total_feats_to_explain = min(args.num_features, dict_model.d_sae)
    print(f"Collecting max-activating examples for first {total_feats_to_explain} features...")
    
    buffer = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        hook_points=[hook_point],
        batch_size=1024,
        buffer_size=4096,
        context_length=256,
        device=device,
    )

    for step in range(8):
        sample_batch = buffer.next_batch()
        if isinstance(sample_batch, (tuple, list)):
            sample_batch = sample_batch[0]
        
        dummy_ids = torch.arange(sample_batch.shape[0], device=device) % tokenizer.vocab_size
        collector.process_sequence_activations(
            input_ids=dummy_ids,
            activations=sample_batch,
            target_features=list(range(total_feats_to_explain))
        )

    w_dec = dict_model.get_decoder_weights()
    os.makedirs(os.path.dirname(args.output_file) or "data", exist_ok=True)
    
    results = {}
    print("\n--- Generating Auto-Interpretation Labels & DLA Metrics ---")
    for feat_id in range(total_feats_to_explain):
        snippets = collector.get_feature_snippets(feat_id)
        base_explanation = explainer.explain_feature(feat_id, snippets)
        
        # Direct Logit Attribution (Top promoted tokens)
        d_v = w_dec[feat_id]
        promoted, suppressed = compute_direct_logit_attribution(
            decoder_vector=d_v,
            unembedding_weights=unembed_weights,
            tokenizer=tokenizer,
            top_k=5,
        )
        
        top_tokens_str = ", ".join([f"'{p['token']}' (+{p['logit']:.1f})" for p in promoted[:3]])
        full_explanation = f"{base_explanation} (Promotes: {top_tokens_str})"

        results[feat_id] = {
            "feature_id": feat_id,
            "layer": hook_point,
            "explanation": full_explanation,
            "snippet_count": len(snippets),
            "top_promoted_tokens": promoted,
            "top_suppressed_tokens": suppressed,
        }
        if feat_id < 20 or feat_id % 20 == 0:
            print(f"Feature #{feat_id:4d} | {full_explanation}")

    save_json(results, args.output_file)
    print(f"\n✅ Saved {len(results)} auto-labeled features to {args.output_file}")


if __name__ == "__main__":
    main()
