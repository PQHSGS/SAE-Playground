from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from src.core.base_dictionary import BaseDictionary
from src.core.hook_manager import HookManager


class FeatureSteeringEngine:
    """
    Real-time feature clamping and steering for causal intervention experiments.
    Injects: h_new = h + alpha * W_dec[feature_id] or clamps f[feature_id] = target_val.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dictionary_model: BaseDictionary,
        hook_point: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dictionary_model = dictionary_model
        self.hook_point = hook_point
        self.hook_manager = HookManager(model)

    def generate_with_steering(
        self,
        prompt: str,
        steered_features: Dict[int, float],  # Dict of {feature_id: steering_coefficient_alpha}
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generates text while applying feature steering vectors to the specified hook point.
        """
        w_dec = self.dictionary_model.get_decoder_weights()
        device = next(self.model.parameters()).device

        # Compute composite steering vector: delta = sum(alpha_i * w_dec[i])
        steering_vec = torch.zeros(self.dictionary_model.d_in, device=device, dtype=torch.float32)
        for feat_id, alpha in steered_features.items():
            if feat_id < w_dec.shape[0]:
                steering_vec += alpha * w_dec[feat_id].to(device, dtype=torch.float32)

        def steering_hook(tensor: torch.Tensor) -> torch.Tensor:
            return tensor + steering_vec.to(dtype=tensor.dtype)

        self.hook_manager.register_intervention_hook(self.hook_point, steering_hook)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        self.hook_manager.remove_hooks()
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
