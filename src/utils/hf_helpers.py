from typing import Tuple, List, Dict
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from dotenv import load_dotenv
from src.utils.logging import setup_logger

load_dotenv()
logger = setup_logger("hf_helpers")


def load_model_and_tokenizer(
    model_name_or_path: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Universal model loader supporting:
    1. Any standard or newly published HuggingFace model (e.g. google/gemma-3-4b-it,
       google/gemma-3-270m, meta-llama/Meta-Llama-3-8B, Qwen/Qwen2.5-7B).
    2. Any custom research checkpoint on HuggingFace Hub or local disk with trust_remote_code=True.
    3. Standalone raw PyTorch architectures (e.g. PlanckGPT) with automatic fallback.
    4. 4-bit / 8-bit quantization via bitsandbytes or native bfloat16 precision.
    """
    logger.info(f"Loading model '{model_name_or_path}' (dtype={torch_dtype}, 4bit={load_in_4bit}, 8bit={load_in_8bit})")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    selected_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # 1. Standard HuggingFace AutoModel pipeline (works for 99.9% of models on HuggingFace)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.cls_token or "<|endoftext|>"

        kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": selected_dtype,
        }
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        if not (load_in_4bit or load_in_8bit):
            kwargs["device_map"] = target_device
        else:
            kwargs["device_map"] = "auto"

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            kwargs["attn_implementation"] = "sdpa"

        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if not (load_in_4bit or load_in_8bit):
            model = model.to(target_device)
        model.eval()
        logger.info(f"Loaded '{model_name_or_path}' via HuggingFace AutoModel successfully on {target_device}.")
        return model, tokenizer

    except Exception as e:
        logger.warning(f"Standard AutoModel load failed ({e}). Checking for custom/raw checkpoint fallbacks...")

        # 2. Transparent fallback for custom repos with raw state_dict (e.g., PlanckGPT)
        if "planckgpt" in model_name_or_path.lower():
            return _load_planckgpt_fallback(model_name_or_path, selected_dtype, device_map)
        
        raise RuntimeError(f"Failed to load model '{model_name_or_path}': {e}")


def _load_planckgpt_fallback(model_name_or_path: str, dtype: torch.dtype, device_map: str):
    from huggingface_hub import hf_hub_download
    from src.utils.planck_gpt import PlanckGPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PlanckGPT(
        vocab_size=50257,
        d_model=896,
        num_layers=14,
        num_heads=7,
        head_dim=128,
        d_ff=3584,
    )

    # Download raw weights from HF
    weight_path = hf_hub_download("cattonpm/PlanckGPT-v0.10.0", "planckgpt.pth")
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Loaded custom PlanckGPT architecture successfully.")
    return model, tokenizer


def get_unembedding_weights(model: nn.Module) -> torch.Tensor:
    """
    Extracts the unembedding matrix W_U of shape (d_model, vocab_size) across diverse model architectures.
    """
    if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        return model.get_output_embeddings().weight.detach().contiguous()
    elif hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        return model.lm_head.weight.detach().contiguous()
    elif hasattr(model, "output") and hasattr(model.output, "weight"):  # PlanckGPT / custom GPT
        return model.output.weight.detach().contiguous()
    elif hasattr(model, "wte") and hasattr(model.wte, "weight"):  # Tied embeddings
        return model.wte.weight.detach().contiguous()
    else:
        for name, param in model.named_parameters():
            if "lm_head" in name or "output" in name or "unembed" in name:
                return param.detach().contiguous()
        raise ValueError(f"Could not automatically locate unembedding matrix W_U in model {model.__class__.__name__}")


def compute_direct_logit_attribution(
    decoder_vector: torch.Tensor,
    unembedding_weights: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    top_k: int = 10,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Computes Direct Logit Attribution (Logit Lens) for a dictionary feature decoder vector:
    Logits = d_i * W_U^T
    Returns top positive (promoted) and top negative (suppressed) vocabulary tokens.
    """
    w_u = unembedding_weights.float().contiguous()
    d_v = decoder_vector.float().to(w_u.device).contiguous()

    if w_u.shape[0] == d_v.shape[0]:
        logits = torch.matmul(d_v, w_u)
    else:
        logits = torch.matmul(w_u, d_v)

    top_pos_vals, top_pos_indices = torch.topk(logits, k=min(top_k, logits.shape[0]))
    top_neg_vals, top_neg_indices = torch.topk(logits, k=min(top_k, logits.shape[0]), largest=False)

    promoted = [
        {"token": tokenizer.decode([idx.item()]), "token_id": idx.item(), "logit": val.item()}
        for idx, val in zip(top_pos_indices, top_pos_vals)
    ]
    suppressed = [
        {"token": tokenizer.decode([idx.item()]), "token_id": idx.item(), "logit": val.item()}
        for idx, val in zip(top_neg_indices, top_neg_vals)
    ]

    return promoted, suppressed
