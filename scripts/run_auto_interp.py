import argparse
import json
import os
import sys
import yaml
import torch
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.hf_helpers import load_model_and_tokenizer, get_unembedding_weights
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
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/gemma3_270m_resid/step_25000")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-3-270m")
    parser.add_argument("--hook_point", type=str, default=None, help="Specific layer or None for all layers")
    parser.add_argument("--all_layers", action="store_true", default=False, help="Process all available layers in checkpoint")
    parser.add_argument("--num_features", type=int, default=5120, help="Number of features to explain per layer")
    parser.add_argument("--output_file", type=str, default="data/gemma3_270m_feature_metadata.json")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu or cuda)")
    parser.add_argument("--explainer_model", type=str, default="google/gemma-3-4b-it", help="HuggingFace model for auto-interp")
    parser.add_argument("--load_in_4bit", action="store_true", default=True, help="Load explainer in 4-bit")
    parser.add_argument("--provider", type=str, default="llm", choices=["llm", "heuristic"], help="Explanation provider (llm or heuristic)")
    return parser.parse_args()


def process_layer_metadata(
    layer_hook: str,
    dict_model,
    unembed_weights: torch.Tensor,
    tokenizer,
    collector: FeatureSampleCollector,
    explainer: FeatureExplainer,
    num_features: int,
):
    d_sae = dict_model.d_sae
    total_feats = min(num_features, d_sae)
    w_dec = dict_model.get_decoder_weights()[:total_feats].float().to("cpu")
    w_u = unembed_weights.float().to("cpu")

    print(f"⚡ Computing Vectorized Direct Logit Attribution for {total_feats} features in '{layer_hook}'...")
    # Fast vectorized Logit Lens: (total_feats, d_in) @ (d_in, vocab_size) -> (total_feats, vocab_size)
    if w_u.shape[0] == w_dec.shape[1]:
        logits = torch.matmul(w_dec, w_u)
    else:
        logits = torch.matmul(w_dec, w_u.T)

    top_pos_vals, top_pos_idx = torch.topk(logits, k=5, dim=-1)
    top_neg_vals, top_neg_idx = torch.topk(logits, k=5, dim=-1, largest=False)

    layer_results = {}
    for feat_id in range(total_feats):
        snippets = collector.get_feature_snippets(feat_id) if collector else []
        base_explanation = explainer.explain_feature(feat_id, snippets) if collector else f"Feature #{feat_id}"

        promoted = [
            {"token": tokenizer.decode([idx.item()]), "token_id": idx.item(), "logit": float(val.item())}
            for idx, val in zip(top_pos_idx[feat_id], top_pos_vals[feat_id])
        ]
        suppressed = [
            {"token": tokenizer.decode([idx.item()]), "token_id": idx.item(), "logit": float(val.item())}
            for idx, val in zip(top_neg_idx[feat_id], top_neg_vals[feat_id])
        ]

        top_tokens_str = ", ".join([f"'{p['token']}' (+{p['logit']:.1f})" for p in promoted[:3]])
        full_explanation = f"{base_explanation} (Promotes: {top_tokens_str})"

        layer_results[feat_id] = {
            "feature_id": feat_id,
            "layer": layer_hook,
            "explanation": full_explanation,
            "snippet_count": len(snippets),
            "top_promoted_tokens": promoted,
            "top_suppressed_tokens": suppressed,
        }

    return layer_results


def main():
    args = parse_args()
    device = args.device

    model_name = args.model_name_or_path
    checkpoint_dir = args.checkpoint_dir

    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory '{checkpoint_dir}' does not exist.")

    # 1. Discover all available layers in checkpoint
    layer_map = {}
    multi_cfg_file = os.path.join(checkpoint_dir, "multi_sae_config.json")
    if os.path.exists(multi_cfg_file):
        with open(multi_cfg_file, "r", encoding="utf-8") as f:
            multi_cfg = json.load(f)
        for hp in multi_cfg.get("hook_points", []):
            subpath = os.path.join(checkpoint_dir, hp.replace(".", "_"))
            if os.path.exists(subpath):
                layer_map[hp] = subpath
    elif os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        hook_name = args.hook_point or "model.layers.9"
        layer_map[hook_name] = checkpoint_dir

    if args.hook_point and not args.all_layers:
        layer_map = {k: v for k, v in layer_map.items() if k == args.hook_point}

    print(f"📦 Loading base LLM '{model_name}' on {device}...")
    model, tokenizer = load_model_and_tokenizer(model_name, device_map=device)
    unembed_weights = get_unembedding_weights(model).to(device)

    explainer = FeatureExplainer(
        model_name=args.explainer_model,
        load_in_4bit=args.load_in_4bit,
        device=device,
        provider=args.provider,
    )
    full_metadata_catalog = {}

    # Check if partial metadata exists to merge
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                full_metadata_catalog = json.load(f)
        except Exception:
            full_metadata_catalog = {}

    print(f"🚀 Starting Auto-Interpretation across {len(layer_map)} layer(s)...")
    for layer_idx, (hook_point, layer_subpath) in enumerate(layer_map.items(), 1):
        print(f"\n[{layer_idx}/{len(layer_map)}] Processing layer '{hook_point}' from {layer_subpath}...")
        
        cfg_file = os.path.join(layer_subpath, "config.json")
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cls_name = cfg.get("class_name", "BatchTopKSAE")
        cls = get_dictionary_cls(cls_name)
        dict_model = cls.from_pretrained(layer_subpath, device=device)

        collector = FeatureSampleCollector(dictionary_model=dict_model, tokenizer=tokenizer)
        
        # Sample activation buffer for token snippets (sample subset to keep execution ultra-fast)
        num_sample_feats = min(200, dict_model.d_sae)
        buffer = ActivationBuffer(
            model=model,
            tokenizer=tokenizer,
            hook_points=[hook_point],
            batch_size=512,
            buffer_size=2048,
            context_length=128,
            device=device,
        )

        for step in range(3):
            sample_batch = buffer.next_batch()
            if isinstance(sample_batch, (tuple, list)):
                sample_batch = sample_batch[0]
            dummy_ids = torch.arange(sample_batch.shape[0], device=device) % tokenizer.vocab_size
            collector.process_sequence_activations(
                input_ids=dummy_ids,
                activations=sample_batch,
                target_features=list(range(num_sample_feats))
            )

        layer_meta = process_layer_metadata(
            layer_hook=hook_point,
            dict_model=dict_model,
            unembed_weights=unembed_weights,
            tokenizer=tokenizer,
            collector=collector,
            explainer=explainer,
            num_features=args.num_features,
        )

        full_metadata_catalog[hook_point] = layer_meta
        os.makedirs(os.path.dirname(args.output_file) or "data", exist_ok=True)
        save_json(full_metadata_catalog, args.output_file)
        print(f"✅ Completed '{hook_point}': {len(layer_meta)} features labeled & saved.")

    print(f"\n🎉 FULL AUTO-INTERPRETATION COMPLETED! Total layers: {len(full_metadata_catalog)}")
    print(f"Metadata written to {args.output_file}")


if __name__ == "__main__":
    main()
