from typing import Optional
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseSAE, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("gated")
@register_dictionary("gated_sae")
class GatedSAE(BaseSAE):
    """
    Gated Sparse Autoencoder (Rajamanoharan et al. 2024).
    Decouples feature detection gate pi(x) from magnitude estimation r(x):
    pi(x) = Heaviside(W_gate (x - b_dec) + b_gate)
    r(x)  = ReLU(W_mag (x - b_dec) + b_mag)
    f(x)  = pi(x) * r(x)
    """

    def __init__(self, d_in: int, d_sae: int, l1_coeff: float = 1e-3, **kwargs):
        super().__init__(d_in=d_in, d_sae=d_sae, l1_coeff=l1_coeff, **kwargs)
        self.l1_coeff = l1_coeff

        # Gating encoder
        self.w_gate = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_gate = nn.Parameter(torch.zeros(d_sae))

        # Magnitude linear scaling
        self.r_mag = nn.Parameter(torch.zeros(d_sae))
        self.b_mag = nn.Parameter(torch.zeros(d_sae))

        # Decoder
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_gate, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        self.normalize_decoder_weights()

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        gate_pre_acts = torch.matmul(x_centered, self.w_gate) + self.b_gate
        pi_gate = (gate_pre_acts > 0).float()
        
        # Magnitude path with learned weight scaling
        mag_pre_acts = torch.matmul(x_centered, self.w_gate * torch.exp(self.r_mag)) + self.b_mag
        r_mag = torch.relu(mag_pre_acts)

        return pi_gate * r_mag

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
        x_centered = x - self.b_dec
        gate_pre_acts = torch.matmul(x_centered, self.w_gate) + self.b_gate
        pi_gate = (gate_pre_acts > 0).float()

        mag_pre_acts = torch.matmul(x_centered, self.w_gate * torch.exp(self.r_mag)) + self.b_mag
        r_mag = torch.relu(mag_pre_acts)
        f = pi_gate * r_mag

        x_hat = self.decode(f)
        mse_loss = nn.functional.mse_loss(x_hat, target)

        # Gating sparsity loss (applied on sigmoid of gate pre-acts)
        gate_prob = torch.sigmoid(gate_pre_acts)
        l1_loss = self.l1_coeff * gate_prob.sum(dim=-1).mean()
        
        # Gating reconstruction auxiliary loss
        gate_f = torch.relu(gate_pre_acts)
        gate_x_hat = torch.matmul(gate_f, self.w_dec.detach()) + self.b_dec
        aux_loss = nn.functional.mse_loss(gate_x_hat, target)

        total_loss = mse_loss + l1_loss + aux_loss

        return DictionaryOutput(
            reconstructed=x_hat,
            feature_acts=f,
            loss=total_loss,
            loss_dict={"mse_loss": mse_loss, "l1_loss": l1_loss, "aux_loss": aux_loss, "total_loss": total_loss},
            extra_dict={"l0": (f > 0).float().sum(dim=-1).mean().item(), "pre_acts": mag_pre_acts}
        )
