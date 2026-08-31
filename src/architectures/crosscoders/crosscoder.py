from typing import Optional
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseCrosscoder, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("crosscoder")
@register_dictionary("multi_layer_crosscoder")
class MultiLayerCrosscoder(BaseCrosscoder):
    """
    Anthropic-style Multi-Layer Crosscoder.
    Encodes activations across L layers [x_0, ..., x_L] into a shared latent space f,
    decoding with layer-specific decoder matrices [W_dec^(0), ..., W_dec^(L)].
    """

    def __init__(
        self,
        n_layers: int,
        d_in: int,
        d_sae: int,
        k: int = 64,
        aux_loss_coeff: float = 1.0 / 32.0,
        **kwargs
    ):
        super().__init__(n_layers=n_layers, d_in=d_in, d_sae=d_sae, k=k, aux_loss_coeff=aux_loss_coeff, **kwargs)
        self.k = k
        self.aux_loss_coeff = aux_loss_coeff

        # Encoder maps concatenated/summed layer representations to shared latents
        # Shape: (n_layers, d_in, d_sae)
        self.w_enc = nn.Parameter(torch.empty(n_layers, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Layer-specific decoders: (n_layers, d_sae, d_in)
        self.w_dec = nn.Parameter(torch.empty(n_layers, d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(n_layers, d_in))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_enc, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        self.normalize_decoder_weights()

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (..., n_layers, d_in)
        Returns:
            f: Shared latent activations of shape (..., d_sae)
        """
        x_centered = x - self.b_dec
        pre_acts = torch.einsum("...li,lid->...d", x_centered, self.w_enc) + self.b_enc
        pre_acts = torch.relu(pre_acts)

        val, idx = torch.topk(pre_acts, k=min(self.k, pre_acts.shape[-1]), dim=-1)
        f = torch.zeros_like(pre_acts)
        f.scatter_(dim=-1, index=idx, src=val)
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: Latent activations of shape (..., d_sae)
        Returns:
            x_hat: Reconstructed layers of shape (..., n_layers, d_in)
        """
        x_hat = torch.einsum("...d,ldi->...li", f, self.w_dec) + self.b_dec
        return x_hat

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        dead_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> DictionaryOutput:
        target = target if target is not None else x
        x_centered = x - self.b_dec
        pre_acts = torch.relu(torch.einsum("...li,lid->...d", x_centered, self.w_enc) + self.b_enc)

        val, idx = torch.topk(pre_acts, k=min(self.k, pre_acts.shape[-1]), dim=-1)
        f = torch.zeros_like(pre_acts)
        f.scatter_(dim=-1, index=idx, src=val)

        x_hat = self.decode(f)
        mse_loss = nn.functional.mse_loss(x_hat, target)

        loss_dict = {"mse_loss": mse_loss}
        total_loss = mse_loss

        # Auxiliary loss for dead latents (OpenAI Aux Loss formulation)
        if dead_mask is not None and dead_mask.any() and self.aux_loss_coeff > 0:
            residual = target - x_hat
            dead_pre_acts = pre_acts * dead_mask.float()
            dead_k = min(self.k, int(dead_mask.sum().item()))
            if dead_k > 0:
                dead_val, dead_idx = torch.topk(dead_pre_acts, k=dead_k, dim=-1)
                dead_f = torch.zeros_like(dead_pre_acts)
                dead_f.scatter_(dim=-1, index=dead_idx, src=dead_val)
                aux_recon = torch.einsum("...d,ldi->...li", dead_f, self.w_dec)
                aux_loss = self.aux_loss_coeff * nn.functional.mse_loss(aux_recon, residual)
                total_loss = total_loss + aux_loss
                loss_dict["aux_loss"] = aux_loss

        loss_dict["total_loss"] = total_loss
        return DictionaryOutput(
            reconstructed=x_hat,
            feature_acts=f,
            loss=total_loss,
            loss_dict=loss_dict,
            extra_dict={"l0": float(self.k), "pre_acts": pre_acts}
        )
