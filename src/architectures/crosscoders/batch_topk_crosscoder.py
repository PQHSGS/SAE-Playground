from typing import Optional
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseCrosscoder, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("batch_topk_crosscoder")
class BatchTopKCrosscoder(BaseCrosscoder):
    """
    BatchTopK Multi-Layer Crosscoder.
    Applies global batch-level top-k selection across all multi-layer tokens
    to eliminate dead features across cross-layer dictionaries.
    """

    def __init__(self, n_layers: int, d_in: int, d_sae: int, k: int = 64, **kwargs):
        super().__init__(n_layers=n_layers, d_in=d_in, d_sae=d_sae, k=k, **kwargs)
        self.k = k

        self.w_enc = nn.Parameter(torch.empty(n_layers, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
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
        x_centered = x - self.b_dec
        pre_acts = torch.einsum("...li,lid->...d", x_centered, self.w_enc) + self.b_enc
        pre_acts = torch.relu(pre_acts)

        orig_shape = pre_acts.shape
        flat = pre_acts.view(-1, self.d_sae)
        batch_tokens = flat.shape[0]

        if self.training:
            total_k = min(batch_tokens * self.k, flat.numel())
            flat_view = flat.view(-1)
            val, idx = torch.topk(flat_view, k=total_k)
            sparse_flat = torch.zeros_like(flat_view)
            sparse_flat.scatter_(dim=0, index=idx, src=val)
            return sparse_flat.view(orig_shape)
        else:
            val, idx = torch.topk(pre_acts, k=min(self.k, self.d_sae), dim=-1)
            f = torch.zeros_like(pre_acts)
            f.scatter_(dim=-1, index=idx, src=val)
            return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...d,ldi->...li", f, self.w_dec) + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        dead_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> DictionaryOutput:
        target = target if target is not None else x
        f = self.encode(x)
        x_hat = self.decode(f)

        mse_loss = nn.functional.mse_loss(x_hat, target)
        return DictionaryOutput(
            reconstructed=x_hat,
            feature_acts=f,
            loss=mse_loss,
            loss_dict={"mse_loss": mse_loss, "total_loss": mse_loss},
            extra_dict={"l0": (f > 0).float().sum(dim=-1).mean().item()}
        )
