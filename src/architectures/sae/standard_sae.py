from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseSAE, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("standard")
@register_dictionary("standard_sae")
@register_dictionary("relu_sae")
class StandardSAE(BaseSAE):
    """
    Standard Sparse Autoencoder with ReLU activation and L1 sparsity penalty.
    f(x) = ReLU(W_enc (x - b_dec) + b_enc)
    x_hat = W_dec f(x) + b_dec
    """

    def __init__(self, d_in: int, d_sae: int, l1_coeff: float = 1e-3, **kwargs):
        super().__init__(d_in=d_in, d_sae=d_sae, l1_coeff=l1_coeff, **kwargs)
        self.l1_coeff = l1_coeff

        self.w_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_enc, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        self.normalize_decoder_weights()

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    def encode(self, x: torch.Tensor, return_pre_acts: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_centered = x - self.b_dec
        pre_acts = torch.matmul(x_centered, self.w_enc) + self.b_enc
        f = torch.relu(pre_acts)
        if return_pre_acts:
            return f, pre_acts
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
        f, pre_acts = self.encode(x, return_pre_acts=True)
        x_hat = self.decode(f)

        # Compute losses according to Bricken et al. 2023 (scaled by decoder norms)
        mse_loss = nn.functional.mse_loss(x_hat, target)
        dec_norms = torch.norm(self.w_dec, dim=-1)
        l1_loss = self.l1_coeff * (f * dec_norms).sum(dim=-1).mean()
        total_loss = mse_loss + l1_loss

        return DictionaryOutput(
            reconstructed=x_hat,
            feature_acts=f,
            loss=total_loss,
            loss_dict={"mse_loss": mse_loss, "l1_loss": l1_loss, "total_loss": total_loss},
            extra_dict={"l0": (f > 0).float().sum(dim=-1).mean().item(), "pre_acts": pre_acts}
        )
