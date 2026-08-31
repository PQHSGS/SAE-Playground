from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class HookConfig(BaseModel):
    """
    Configuration for hooking target LLM submodules (Residual, Attention, or MLP).
    """
    model_name_or_path: str = Field(..., description="HuggingFace model ID or local directory")
    hook_points: List[str] = Field(..., description="List of submodule dot-paths to hook (Residual, MLP, or Attention)")
    target_hook_points: Optional[List[str]] = Field(None, description="For transcoders, target submodules (e.g. ['model.layers.4.mlp.down_proj'])")
    torch_dtype: str = Field("bfloat16", description="Model loading dtype: 'bfloat16', 'float32', 'float16'")
    device_map: str = Field("auto", description="Device placement: 'auto', 'cuda', 'cpu'")
    load_in_8bit: bool = Field(False, description="Load model in 8-bit quantization")
    load_in_4bit: bool = Field(False, description="Load model in 4-bit quantization")
    trust_remote_code: bool = Field(True, description="Allow custom HuggingFace remote code")
    context_length: int = Field(1024, description="Context sequence length for tokenization")
    mask_bos: bool = Field(True, description="Mask out the first token (BOS / attention sink)")
    normalize_activations: bool = Field(True, description="Normalize activations with running mean & norm")


class DictionaryConfig(BaseModel):
    """
    Configuration for Sparse Autoencoder / Transcoder / Crosscoder architectures.
    """
    architecture: str = Field("topk", description="Architecture name: 'topk', 'batch_topk', 'jumprelu', 'gated', 'standard', 'treesae', 'sasa', 'matryoshka', 'matching_pursuit', 'skip_transcoder', 'crosscoder'")
    d_in: int = Field(..., description="Input activation dimension")
    d_sae: int = Field(..., description="Dictionary latent dimension (e.g. d_in * expansion_factor)")
    d_out: Optional[int] = Field(None, description="Output dimension (for transcoders)")
    expansion_factor: Optional[int] = Field(16, description="Expansion multiplier over d_in")
    
    # Architecture-specific hyperparams
    k: Optional[int] = Field(32, description="TopK / BatchTopK active latent count")
    l1_coeff: Optional[float] = Field(1e-3, description="L1 sparsity coefficient for standard/gated SAE")
    jumprelu_init_threshold: Optional[float] = Field(0.001, description="Initial threshold for JumpReLU")
    jumprelu_bandwidth: Optional[float] = Field(0.001, description="Bandwidth for JumpReLU STE gradient")
    aux_loss_coeff: Optional[float] = Field(1.0/32.0, description="OpenAI TopK auxiliary loss coefficient for dead latents")
    tree_depth: Optional[int] = Field(3, description="Tree depth for TreeSAE")
    tree_branching: Optional[int] = Field(4, description="Branching factor for TreeSAE")
    n_layers: Optional[int] = Field(None, description="Number of layers for Crosscoder")
    extra_params: Dict[str, Any] = Field(default_factory=dict, description="Additional custom architecture params")


class TrainingConfig(BaseModel):
    """
    Configuration for training pipeline.
    """
    dataset_path: str = Field("HuggingFaceFW/fineweb-edu", description="HuggingFace dataset path or local JSON/Parquet")
    dataset_name: Optional[str] = Field("sample-10BT", description="Dataset config name / subset")
    dataset_split: str = Field("train", description="Dataset split")
    batch_size: int = Field(4096, description="Token-level batch size for dictionary training")
    learning_rate: float = Field(3e-4, description="Peak learning rate for AdamW")
    min_learning_rate: float = Field(1e-5, description="Final learning rate after cosine decay")
    lr_warmup_steps: int = Field(1000, description="Number of warmup steps for learning rate")
    total_steps: int = Field(50000, description="Total training steps")
    checkpoint_steps: int = Field(5000, description="Save checkpoint frequency in steps")
    eval_steps: int = Field(1000, description="Evaluation frequency in steps")
    
    # Dead feature & Ghost grad handling
    dead_feature_threshold: int = Field(1_000_000, description="Token count without activation before a feature is marked dead")
    use_ghost_grads: bool = Field(True, description="Enable Ghost Gradients auxiliary residual pass for dead features")
    ghost_grad_coeff: float = Field(0.1, description="Ghost gradient scaling factor")
    
    # WandB & Checkpoint paths
    wandb_project: Optional[str] = Field("interpret-sae-playground", description="WandB project name")
    wandb_run_name: Optional[str] = Field(None, description="WandB run name")
    wandb_entity: Optional[str] = Field(None, description="WandB entity")
    output_dir: str = Field("checkpoints", description="Directory to save checkpoints")
    seed: int = Field(42, description="Random seed")


class AutoInterpConfig(BaseModel):
    """
    Configuration for 100% Local Auto-Interpretation & Simulation Scoring.
    """
    llm_model_name: str = Field("google/gemma-3-4b-it", description="Local instruct LLM for explanation generation")
    load_in_4bit: bool = Field(True, description="Load local explainer LLM in 4-bit quantization")
    load_in_8bit: bool = Field(False, description="Load local explainer LLM in 8-bit quantization")
    torch_dtype: str = Field("bfloat16", description="Local explainer dtype")
    max_activating_samples: int = Field(20, description="Number of top max-activating text snippets per feature")
    quantile_samples: int = Field(10, description="Number of quantile-distributed text snippets")
    context_tokens_window: int = Field(64, description="Window size around the max activating token")
    scoring_samples: int = Field(20, description="Number of held-out snippets for simulation scoring")
    output_dir: str = Field("data/auto_interp", description="Output directory for generated explanations")
