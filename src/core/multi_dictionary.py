import os
import json
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseDictionary, DictionaryOutput
from src.architectures.registry import get_dictionary_cls


class MultiLayerDictionary(nn.Module):
    """
    Ensemble container holding independent dictionary models across multiple model layers.
    Allows training M separate layer SAEs simultaneously from a single LLM forward pass.
    
    Key Features:
    1. OOP Cleanliness: Each layer is an independent BaseDictionary instance.
    2. Independent Checkpointing: Each layer is saved to its own subfolder, loadable individually.
    3. Fault-Tolerant: Independent forward/backward passes per layer.
    """

    def __init__(self, dictionaries: Dict[str, BaseDictionary]):
        super().__init__()
        self.dictionaries = nn.ModuleDict({
            self._sanitize_key(k): v for k, v in dictionaries.items()
        })
        self.hook_point_map = {self._sanitize_key(k): k for k in dictionaries.keys()}
        self.reverse_map = {k: self._sanitize_key(k) for k in dictionaries.keys()}

    @staticmethod
    def _sanitize_key(key: str) -> str:
        """ModuleDict keys cannot contain dots."""
        return key.replace(".", "_")

    def get_dictionary(self, hook_point: str) -> BaseDictionary:
        key = self.reverse_map.get(hook_point, self._sanitize_key(hook_point))
        return self.dictionaries[key]

    def encode(self, activations_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        features = {}
        for hp, act in activations_dict.items():
            key = self.reverse_map.get(hp, self._sanitize_key(hp))
            if key in self.dictionaries:
                features[hp] = self.dictionaries[key].encode(act)
        return features

    def decode(self, features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reconstructions = {}
        for hp, feat in features_dict.items():
            key = self.reverse_map.get(hp, self._sanitize_key(hp))
            if key in self.dictionaries:
                reconstructions[hp] = self.dictionaries[key].decode(feat)
        return reconstructions

    def forward(
        self,
        activations_dict: Dict[str, torch.Tensor],
        targets_dict: Optional[Dict[str, torch.Tensor]] = None,
        dead_masks_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, DictionaryOutput]]:
        """
        Execute forward passes across all layer dictionaries.
        Returns:
            total_loss: Sum of losses across all layers.
            outputs: Dictionary mapping hook_point to DictionaryOutput.
        """
        outputs = {}
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        targets_dict = targets_dict or activations_dict
        dead_masks_dict = dead_masks_dict or {}

        for hp, act in activations_dict.items():
            key = self.reverse_map.get(hp, self._sanitize_key(hp))
            if key in self.dictionaries:
                target = targets_dict.get(hp, act)
                dead_mask = dead_masks_dict.get(hp, None)
                out = self.dictionaries[key](act, target=target, dead_mask=dead_mask)
                outputs[hp] = out
                total_loss = total_loss + out.loss

        return total_loss, outputs

    def normalize_decoder_weights(self, eps: float = 1e-8) -> None:
        for sae in self.dictionaries.values():
            sae.normalize_decoder_weights(eps=eps)

    def __len__(self) -> int:
        return len(self.dictionaries)

    def __getitem__(self, hook_point: str) -> BaseDictionary:
        return self.get_dictionary(hook_point)

    @property
    def d_in(self) -> int:
        return next(iter(self.dictionaries.values())).d_in

    @property
    def d_sae(self) -> int:
        return next(iter(self.dictionaries.values())).d_sae

    @property
    def d_out(self) -> int:
        return next(iter(self.dictionaries.values())).d_out

    def save_pretrained(self, save_dir: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Saves each layer dictionary into its own independent subdirectory:
        save_dir/
          multi_sae_config.json (top-level ensemble manifest)
          layer_0/ (standard standalone BaseDictionary save folder)
          layer_1/ ...
        
        If len(dictionaries) == 1, also saves directly to save_dir for 100% single-SAE compatibility.
        """
        os.makedirs(save_dir, exist_ok=True)
        manifest = {
            "hook_points": list(self.hook_point_map.values()),
            "sanitized_keys": list(self.hook_point_map.keys()),
            "metadata": metadata or {},
        }
        with open(os.path.join(save_dir, "multi_sae_config.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        for sanitized_key, sae in self.dictionaries.items():
            layer_dir = os.path.join(save_dir, sanitized_key)
            sae.save_pretrained(layer_dir, metadata={"original_hook_point": self.hook_point_map[sanitized_key]})

        # For single-layer special case (N=1), also save directly to root save_dir
        if len(self.dictionaries) == 1:
            single_sae = next(iter(self.dictionaries.values()))
            single_sae.save_pretrained(save_dir, metadata=metadata)

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu", dtype: torch.dtype = torch.float32) -> "MultiLayerDictionary":
        with open(os.path.join(load_dir, "multi_sae_config.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)

        dict_map = {}
        for sanitized_key in manifest["sanitized_keys"]:
            layer_dir = os.path.join(load_dir, sanitized_key)
            with open(os.path.join(layer_dir, "config.json"), "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cls_name = cfg.get("class_name", "TopKSAE")
            sae_cls = get_dictionary_cls(cls_name)
            sae_instance = sae_cls.from_pretrained(layer_dir, device=device, dtype=dtype)
            orig_hook = cfg.get("metadata", {}).get("original_hook_point", sanitized_key)
            dict_map[orig_hook] = sae_instance

        return cls(dict_map)
