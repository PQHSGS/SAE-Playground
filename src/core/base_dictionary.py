import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


@dataclass
class DictionaryOutput:
    """
    Standardized output dataclass for all dictionary architectures.
    """
    reconstructed: torch.Tensor             # Shape: (..., d_out)
    feature_acts: torch.Tensor              # Shape: (..., d_sae)
    loss: Optional[torch.Tensor] = None     # Scalar total loss
    loss_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    extra_dict: Dict[str, Any] = field(default_factory=dict)


class BaseDictionary(nn.Module, ABC):
    """
    Abstract Base Class for all dictionary representations (SAEs, Transcoders, Crosscoders).
    """

    def __init__(self, d_in: int, d_sae: int, d_out: Optional[int] = None, **kwargs):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.d_out = d_out or d_in
        self.config_kwargs = kwargs

    @abstractmethod
    def encode(self, x: torch.Tensor, return_pre_acts: bool = False) -> Any:
        """
        Map input activations to sparse latent feature activations.
        Args:
            x: Input tensor of shape (..., d_in)
            return_pre_acts: If True, returns (f, pre_acts) or (f, ...)
        Returns:
            f: Sparse feature activations of shape (..., d_sae) or tuple with pre-activations
        """

    @abstractmethod
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct output activations from sparse latent feature activations.
        Args:
            f: Sparse feature activations of shape (..., d_sae)
        Returns:
            x_hat: Reconstructed tensor of shape (..., d_out)
        """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        dead_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> DictionaryOutput:
        """
        Execute forward pass and compute relevant losses.
        """

    @abstractmethod
    def get_decoder_weights(self) -> torch.Tensor:
        """
        Return the primary decoder weight matrix W_dec of shape (d_sae, d_out).
        """

    @torch.no_grad()
    def normalize_decoder_weights(self, eps: float = 1e-8) -> None:
        """
        Project decoder columns/rows to unit norm: ||W_dec[i]||_2 = 1.
        Essential for stable sparsity and preventing feature shrinkage collapse.
        """
        w_dec = self.get_decoder_weights()
        if w_dec.ndim == 2:
            norms = torch.norm(w_dec, p=2, dim=1, keepdim=True) + eps
            w_dec.div_(norms)
        elif w_dec.ndim == 3:  # Crosscoder: (L, d_sae, d_out)
            norms = torch.norm(w_dec, p=2, dim=-1, keepdim=True) + eps
            w_dec.div_(norms)

    def save_pretrained(self, save_dir: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model weights as safetensors and architecture configuration as json.
        """
        os.makedirs(save_dir, exist_ok=True)
        config_dict = {
            "class_name": self.__class__.__name__,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "d_out": self.d_out,
            "config_kwargs": self.config_kwargs,
            "metadata": metadata or {},
        }
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)

        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state, os.path.join(save_dir, "model.safetensors"))

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> "BaseDictionary":
        """
        Load dictionary model from directory.
        """
        with open(os.path.join(load_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        kwargs = config.get("config_kwargs", {}).copy()
        d_out = config.get("d_out", config["d_in"])
        kwargs.pop("d_out", None)

        instance = cls(
            d_in=config["d_in"],
            d_sae=config["d_sae"],
            d_out=d_out,
            **kwargs
        )
        state_dict = load_file(os.path.join(load_dir, "model.safetensors"), device="cpu")
        instance.load_state_dict(state_dict)
        instance.to(device=device, dtype=dtype)
        return instance


class BaseSAE(BaseDictionary):
    """
    Base Class for standard Autoencoders where input and target space are identical (x -> f -> x_hat).
    """
    def __init__(self, d_in: int, d_sae: int, **kwargs):
        kwargs.pop("d_out", None)
        super().__init__(d_in=d_in, d_sae=d_sae, d_out=d_in, **kwargs)


class BaseTranscoder(BaseDictionary):
    """
    Base Class for Transcoders where input x_l maps to target y_l or x_{l+1}.
    """
    def __init__(self, d_in: int, d_sae: int, d_out: Optional[int] = None, **kwargs):
        super().__init__(d_in=d_in, d_sae=d_sae, d_out=d_out or d_in, **kwargs)


class BaseCrosscoder(BaseDictionary):
    """
    Base Class for Multi-Layer Crosscoders mapping activations across L layers [x_0, ..., x_L]
    to shared sparse latents f, and decoding to [x_hat_0, ..., x_hat_L].
    """
    def __init__(self, n_layers: int, d_in: int, d_sae: int, **kwargs):
        super().__init__(d_in=d_in, d_sae=d_sae, d_out=d_in, **kwargs)
        self.n_layers = n_layers
