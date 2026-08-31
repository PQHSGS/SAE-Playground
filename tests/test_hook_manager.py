import pytest
import torch
import torch.nn as nn
from src.core.hook_manager import HookManager


class ToyTransformerBlock(nn.Module):
    def __init__(self, d_model=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

    def forward(self, x):
        return x + self.mlp(x)


class ToyLanguageModel(nn.Module):
    def __init__(self, d_model=32, n_layers=2):
        super().__init__()
        self.blocks = nn.ModuleList([ToyTransformerBlock(d_model) for _ in range(n_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_hook_manager_extraction_and_intervention():
    model = ToyLanguageModel()
    hook_mgr = HookManager(model)

    # 1. Test listing hookable modules
    modules = hook_mgr.list_hookable_modules(pattern=r"mlp")
    assert len(modules) >= 2

    # 2. Test forward hooks
    hook_mgr.register_forward_hooks(["blocks.0.mlp", "blocks.1.mlp"])
    x = torch.randn(2, 8, 32)
    _ = model(x)

    assert "blocks.0.mlp" in hook_mgr.activations
    assert hook_mgr.activations["blocks.0.mlp"].shape == (2, 8, 32)
    hook_mgr.remove_hooks()
    assert len(hook_mgr.activations) == 0

    # 3. Test intervention hook
    def zero_hook(tensor):
        return torch.zeros_like(tensor)

    hook_mgr.register_intervention_hook("blocks.0.mlp", zero_hook)
    out_steered = model(x)
    hook_mgr.remove_hooks()
    out_normal = model(x)

    assert not torch.allclose(out_steered, out_normal)
