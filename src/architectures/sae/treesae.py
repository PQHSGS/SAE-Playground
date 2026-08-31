from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseSAE, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("treesae")
@register_dictionary("tree_sae")
@register_dictionary("hierarchical_sae")
class TreeSAE(BaseSAE):
    """
    TreeSAE (Hierarchical Tree-Structured Sparse Autoencoder).
    Features are organized in a tree hierarchy from coarse concept nodes to fine leaf nodes.
    Routing dynamically navigates from root concepts down to specific sub-features.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        tree_depth: int = 3,
        tree_branching: int = 4,
        k: int = 16,
        **kwargs
    ):
        super().__init__(d_in=d_in, d_sae=d_sae, tree_depth=tree_depth, tree_branching=tree_branching, k=k, **kwargs)
        self.tree_depth = tree_depth
        self.tree_branching = tree_branching
        self.k = k

        # Root coarse encoder and multi-level branch encoders
        self.coarse_enc = nn.Linear(d_in, d_sae // 4, bias=True)
        self.fine_enc = nn.Linear(d_in + (d_sae // 4), d_sae, bias=True)

        self.w_dec_coarse = nn.Parameter(torch.empty(d_sae // 4, d_in))
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.coarse_enc.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.fine_enc.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_dec_coarse, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        self.normalize_decoder_weights()

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    def encode(self, x: torch.Tensor, return_pre_acts: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x_centered = x - self.b_dec
        # Stage 1: Coarse routing
        coarse_acts = torch.relu(self.coarse_enc(x_centered))
        
        # Stage 2: Fine leaf activations conditioned on coarse concepts
        joint_input = torch.cat([x_centered, coarse_acts], dim=-1)
        fine_pre_acts = torch.relu(self.fine_enc(joint_input))

        # Top-K selection on fine features
        val, idx = torch.topk(fine_pre_acts, k=min(self.k, fine_pre_acts.shape[-1]), dim=-1)
        f = torch.zeros_like(fine_pre_acts)
        f.scatter_(dim=-1, index=idx, src=val)

        if return_pre_acts:
            return f, coarse_acts, fine_pre_acts
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.matmul(f, self.w_dec) + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        dead_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> DictionaryOutput:
        target = target if target is not None else x
        f, coarse_acts, fine_pre_acts = self.encode(x, return_pre_acts=True)
        x_hat = self.decode(f)
        mse_fine = nn.functional.mse_loss(x_hat, target)

        # Hierarchical coarse reconstruction auxiliary loss
        coarse_hat = torch.matmul(coarse_acts, self.w_dec_coarse) + self.b_dec
        mse_coarse = nn.functional.mse_loss(coarse_hat, target)
        total_loss = mse_fine + 0.25 * mse_coarse

        return DictionaryOutput(
            reconstructed=x_hat,
            feature_acts=f,
            loss=total_loss,
            loss_dict={"mse_loss": mse_fine, "mse_coarse": mse_coarse, "total_loss": total_loss},
            extra_dict={"l0": float(self.k), "pre_acts": fine_pre_acts}
        )
