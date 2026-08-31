from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseTranscoder, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("gated_transcoder")
class GatedTranscoder(BaseTranscoder):
    """
    Gated Transcoder (x_in -> pi_gate * r_mag -> y_out).
    Decouples gating detection from magnitude estimation for transcoders.
    """

    def __init__(self, d_in: int, d_sae: int, d_out: Optional[int] = None, l1_coeff: float = 1e-3, **kwargs):
        d_out = d_out or d_in
        super().__init__(d_in=d_in, d_sae=d_sae, d_out=d_out, l1_coeff=l1_coeff, **kwargs)
        self.l1_coeff = l1_coeff

        self.w_gate = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_gate = nn.Parameter(torch.zeros(d_sae))
        self.r_mag = nn.Parameter(torch.zeros(d_sae))
        self.b_mag = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_out))
        self.b_dec = nn.Parameter(torch.zeros(d_out))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_gate, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        self.normalize_decoder_weights()

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        gate_pre = torch.matmul(x, self.w_gate) + self.b_gate
        pi_gate = (gate_pre > 0).float()
        mag_pre = torch.matmul(x, self.w_gate * torch.exp(self.r_mag)) + self.b_mag
        return pi_gate * torch.relu(mag_pre)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.matmul(f, self.w_dec) + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        dead_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> DictionaryOutput:
        if target is None:
            raise ValueError("GatedTranscoder requires a target tensor.")

        gate_pre = torch.matmul(x, self.w_gate) + self.b_gate
        pi_gate = (gate_pre > 0).float()
        mag_pre = torch.matmul(x, self.w_gate * torch.exp(self.r_mag)) + self.b_mag
        r_mag = torch.relu(mag_pre)
        f = pi_gate * r_mag

        y_hat = self.decode(f)
        mse_loss = nn.functional.mse_loss(y_hat, target)
        l1_loss = self.l1_coeff * torch.sigmoid(gate_pre).sum(dim=-1).mean()
        total_loss = mse_loss + l1_loss

        return DictionaryOutput(
            reconstructed=y_hat,
            feature_acts=f,
            loss=total_loss,
            loss_dict={"mse_loss": mse_loss, "l1_loss": l1_loss, "total_loss": total_loss},
            extra_dict={"l0": (f > 0).float().sum(dim=-1).mean().item()}
        )
