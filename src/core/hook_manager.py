import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def extract_tensor_from_output(output: Any) -> torch.Tensor:
    """
    Robustly extract the primary activation tensor from varied module output formats:
    - Raw torch.Tensor: (batch, seq_len, dim)
    - Tuple / List: (tensor, ...) -> tensor
    - HuggingFace ModelOutput / Dataclass: extracts .hidden_states or first tensor field
    """
    if isinstance(output, torch.Tensor):
        return output
    elif isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
        raise ValueError(f"Could not find a valid torch.Tensor in tuple/list output of type {[type(x) for x in output]}")
    elif hasattr(output, "hidden_states") and output.hidden_states is not None:
        if isinstance(output.hidden_states, (tuple, list)):
            return output.hidden_states[-1]
        return output.hidden_states
    elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    else:
        # Check dict-like or dataclass-like attributes
        if hasattr(output, "__dict__"):
            for val in output.__dict__.values():
                if isinstance(val, torch.Tensor):
                    return val
        raise TypeError(f"Unsupported module output type: {type(output)}")


class HookManager:
    """
    Universal, architecture-agnostic forward hook and intervention manager for PyTorch LLMs.
    Supports standard HuggingFace models, custom research models (e.g. PlanckGPT), and TransformerLens.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.intervention_hooks: Dict[str, Callable] = {}

    def list_hookable_modules(self, pattern: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        List all hookable submodules in the model, with optional regex filtering.
        Returns:
            List of tuples (submodule_path, class_name)
        """
        results = []
        for name, mod in self.model.named_modules():
            if not name:  # Root module
                continue
            if pattern is None or re.search(pattern, name):
                results.append((name, mod.__class__.__name__))
        return results

    def get_submodule(self, path: str) -> nn.Module:
        """
        Retrieve submodule by dot-separated path (e.g., 'model.layers.4.mlp').
        """
        submod = self.model
        for part in path.split("."):
            if not hasattr(submod, part):
                raise AttributeError(f"Module '{submod.__class__.__name__}' has no attribute '{part}' (full path: {path})")
            submod = getattr(submod, part)
        return submod

    def register_forward_hooks(
        self,
        hook_points: List[str],
        custom_callback: Optional[Callable[[str, torch.Tensor], None]] = None,
    ) -> None:
        """
        Register forward hooks on specified submodule dot-paths.
        Args:
            hook_points: List of submodule paths (e.g., ['model.layers.2.mlp', 'transformer.h.0'])
            custom_callback: Optional callback receiving (hook_point_name, activation_tensor)
        """
        self.remove_hooks()
        self.activations.clear()

        for name in hook_points:
            module = self.get_submodule(name)

            def make_hook(hook_name: str):
                def hook_fn(module, module_in, module_out):
                    act = extract_tensor_from_output(module_out)
                    self.activations[hook_name] = act
                    if custom_callback is not None:
                        custom_callback(hook_name, act)
                return hook_fn

            handle = module.register_forward_hook(make_hook(name))
            self.handles.append(handle)

    def register_intervention_hook(
        self,
        hook_point: str,
        intervention_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """
        Register a forward hook that modifies/clamps the submodule output in-place (e.g. for feature steering).
        intervention_fn receives activation tensor of shape (batch, seq, dim) and returns modified tensor.
        """
        module = self.get_submodule(hook_point)

        def hook_fn(module, module_in, module_out):
            if isinstance(module_out, torch.Tensor):
                return intervention_fn(module_out)
            elif isinstance(module_out, tuple):
                # Replace the primary tensor element in the tuple
                new_list = list(module_out)
                for idx, item in enumerate(new_list):
                    if isinstance(item, torch.Tensor):
                        new_list[idx] = intervention_fn(item)
                        break
                return tuple(new_list)
            else:
                return intervention_fn(module_out)

        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def remove_hooks(self) -> None:
        """
        Remove all active forward and intervention hooks from the model.
        """
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
