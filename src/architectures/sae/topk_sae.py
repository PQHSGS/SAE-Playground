from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseSAE, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("topk")
@register_dictionary("topk_sae")
class TopKSAE(BaseSAE):
    """
    TopK Sparse Autoencoder (Gao et al. / OpenAI 2024).
    Keeps only the top-k largest latent activations per token and zeros out the rest.
    Includes OpenAI Auxiliary Loss for dead latent revival.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int = 32,
        aux_loss_coeff: float = 1.0 / 32.0,
        **kwargs
    ):
        super().__init__(d_in=d_in, d_sae=d_sae, k=k, aux_loss_coeff=aux_loss_coeff, **kwargs)
        self.k = k
        self.aux_loss_coeff = aux_loss_coeff

        self.w_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        self.normalize_decoder_weights()
        self.w_enc.data.copy_(self.w_dec.data.T)
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    def encode(self, x: torch.Tensor, return_pre_acts: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_centered = x - self.b_dec
        raw_pre_acts = torch.matmul(x_centered, self.w_enc) + self.b_enc
        pre_acts = torch.relu(raw_pre_acts)
        
        # Apply exact TopK sparsity per token
        val, idx = torch.topk(pre_acts, k=min(self.k, pre_acts.shape[-1]), dim=-1)
        f = torch.zeros_like(pre_acts)
        f.scatter_(dim=-1, index=idx, src=val)
        if return_pre_acts:
            return f, raw_pre_acts
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
        f, raw_pre_acts = self.encode(x, return_pre_acts=True)

        x_hat = self.decode(f)
        mse_loss = nn.functional.mse_loss(x_hat, target)

        loss_dict = {"mse_loss": mse_loss}
        total_loss = mse_loss

        # Auxiliary loss for dead latents (OpenAI TopK Aux Loss with softplus gradient flow)
        if dead_mask is not None and dead_mask.any() and self.aux_loss_coeff > 0:
            residual = target - x_hat
            dead_pre_acts = torch.nn.functional.softplus(raw_pre_acts) * dead_mask.float()
            dead_k = min(self.k, int(dead_mask.sum().item()))
            if dead_k > 0:
                dead_val, dead_idx = torch.topk(dead_pre_acts, k=dead_k, dim=-1)
                dead_f = torch.zeros_like(dead_pre_acts)
                dead_f.scatter_(dim=-1, index=dead_idx, src=dead_val)
                aux_recon = torch.matmul(dead_f, self.w_dec)
                aux_loss = self.aux_loss_coeff * nn.functional.mse_loss(aux_recon, residual)
                total_loss = total_loss + aux_loss
                loss_dict["aux_loss"] = aux_loss

        loss_dict["total_loss"] = total_loss
        return DictionaryOutput(
            reconstructed=x_hat,
            feature_acts=f,
            loss=total_loss,
            loss_dict=loss_dict,
            extra_dict={"l0": float(self.k), "pre_acts": raw_pre_acts}
        )
