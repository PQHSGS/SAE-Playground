import argparse
import os
import sys
import yaml
import torch
from dotenv import load_dotenv

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.config import TrainingConfig
from src.core.activation_buffer import ActivationBuffer
from src.architectures.registry import build_dictionary
from src.training.trainer import DictionaryTrainer
from src.utils.hf_helpers import load_model_and_tokenizer
from src.utils.logging import setup_logger

load_dotenv()
logger = setup_logger("train")


def load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoders, Transcoders, or Crosscoders via YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if not cfg:
        raise FileNotFoundError(f"Config file '{args.config}' was not found or is empty.")

    # Extract configurations directly from YAML
    model_name = cfg.get("model_name_or_path", "gpt2")
    hook_points = cfg.get("hook_points", ["transformer.h.6"])
    target_hook_points = cfg.get("target_hook_points", None)
    torch_dtype = cfg.get("torch_dtype", "bfloat16")
    load_in_4bit = cfg.get("load_in_4bit", False)
    load_in_8bit = cfg.get("load_in_8bit", False)

    arch_name = cfg.get("architecture", "topk")
    expansion = cfg.get("expansion_factor", 16)
    k = cfg.get("k", 32)
    l1_coeff = cfg.get("l1_coeff", 1e-3)

    dataset_path = cfg.get("dataset_path", "HuggingFaceFW/fineweb-edu")
    dataset_name = cfg.get("dataset_name", "sample-10BT")
    cfg.get("dataset_split", "train")
    cfg.get("context_length", 1024)
    cfg.get("mask_bos", True)

    batch_size = cfg.get("batch_size", 4096)
    lr = cfg.get("learning_rate", 3e-4)
    min_lr = cfg.get("min_learning_rate", 1e-5)
    warmup_steps = cfg.get("lr_warmup_steps", 1000)
    total_steps = cfg.get("total_steps", 5000)
    ckpt_steps = cfg.get("checkpoint_steps", 2500)
    output_dir = cfg.get("output_dir", "checkpoints")
    wandb_project = cfg.get("wandb_project", "interpret-sae-playground")
    wandb_run_name = cfg.get("wandb_run_name", None)
    use_ghost_grads = cfg.get("use_ghost_grads", True)
    ghost_grad_coeff = cfg.get("ghost_grad_coeff", 0.1)
    seed = cfg.get("seed", 42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target Model: '{model_name}' | Hook Points: {hook_points} | Architecture: '{arch_name}'")

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_name,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        trust_remote_code=True,
    )

    # Probe activation dimension d_in
    probe_buffer = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        hook_points=hook_points,
        target_hook_points=target_hook_points,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        batch_size=min(batch_size, 512),
        buffer_size=1024,
        device=device,
    )
    sample_batch = probe_buffer.next_batch()
    if isinstance(sample_batch, (tuple, list)):
        d_in = sample_batch[0].shape[-1]
        d_out = sample_batch[1].shape[-1]
    else:
        d_in = sample_batch.shape[-1]
        d_out = d_in

    d_sae = d_in * expansion
    logger.info(f"Initialized {arch_name.upper()}: d_in={d_in}, d_sae={d_sae}, d_out={d_out}")

    # Build dictionary model
    kwargs = {
        "k": k,
        "l1_coeff": l1_coeff,
        "d_out": d_out,
        **{k_: v_ for k_, v_ in cfg.items() if k_ not in [
            "architecture", "expansion_factor", "model_name_or_path", "hook_points",
            "target_hook_points", "torch_dtype", "load_in_4bit", "load_in_8bit",
            "dataset_path", "dataset_name", "dataset_split", "batch_size", "learning_rate",
            "min_learning_rate", "lr_warmup_steps", "total_steps", "checkpoint_steps",
            "output_dir", "wandb_project", "seed"
        ]}
    }
    # Check if multi-SAE mode (multiple hook points for standard SAE architectures)
    len(hook_points) > 1 and arch_name not in ["crosscoder", "batch_topk_crosscoder"]

    # Training configuration
    training_cfg = TrainingConfig(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        learning_rate=lr,
        min_learning_rate=min_lr,
        lr_warmup_steps=warmup_steps,
        total_steps=total_steps,
        checkpoint_steps=ckpt_steps,
        output_dir=output_dir,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        use_ghost_grads=use_ghost_grads,
        ghost_grad_coeff=ghost_grad_coeff,
        seed=seed,
    )

    # Build dictionary model collection (N >= 1)
    from src.core.multi_dictionary import MultiLayerDictionary

    if arch_name in ["crosscoder", "batch_topk_crosscoder"]:
        kwargs["n_layers"] = len(hook_points)
        dict_model = MultiLayerDictionary({
            "crosscoder": build_dictionary(architecture=arch_name, d_in=d_in, d_sae=d_sae, **kwargs)
        })
        return_dict = False
    else:
        dict_map = {
            hp: build_dictionary(architecture=arch_name, d_in=d_in, d_sae=d_sae, **kwargs)
            for hp in hook_points
        }
        dict_model = MultiLayerDictionary(dict_map)
        return_dict = True

    full_buffer = ActivationBuffer(
        model=model,
        tokenizer=tokenizer,
        hook_points=hook_points,
        target_hook_points=target_hook_points,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        batch_size=batch_size,
        buffer_size=min(65536, batch_size * 16),
        device=device,
        return_dict=return_dict,
    )

    trainer = DictionaryTrainer(
        dictionary_model=dict_model,
        activation_buffer=full_buffer,
        config=training_cfg,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
