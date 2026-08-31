from typing import List, Optional
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseSAE, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("matryoshka")
@register_dictionary("matryoshka_sae")
@register_dictionary("multiscale_sae")
class MatryoshkaSAE(BaseSAE):
    """
    Matryoshka / Multi-Scale Sparse Autoencoder (Bussmann et al. / Anthropic).
    Nested dictionary representations where nested sub-slices of features 
    [d_1, d_2, ..., d_M] (e.g. [512, 1024, 2048, 4096]) are jointly trained 
    with a multi-prefix reconstruction loss. Enables variable-compute adaptive interpretability.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int = 32,
        prefix_dims: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__(d_in=d_in, d_sae=d_sae, k=k, **kwargs)
        self.k = k
        self.prefix_dims = prefix_dims or [d_sae // 8, d_sae // 4, d_sae // 2, d_sae]
        self.prefix_dims = sorted(list(set(self.prefix_dims)))

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

    def encode(self, x: torch.Tensor, max_prefix: Optional[int] = None) -> torch.Tensor:
        x_centered = x - self.b_dec
        dim_limit = max_prefix or self.d_sae
        w_enc_sub = self.w_enc[:, :dim_limit]
        b_enc_sub = self.b_enc[:dim_limit]

        pre_acts = torch.relu(torch.matmul(x_centered, w_enc_sub) + b_enc_sub)
        val, idx = torch.topk(pre_acts, k=min(self.k, dim_limit), dim=-1)
        f = torch.zeros_like(pre_acts)
        f.scatter_(dim=-1, index=idx, src=val)
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        dim_used = f.shape[-1]
        w_dec_sub = self.w_dec[:dim_used, :]
        return torch.matmul(f, w_dec_sub) + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        dead_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> DictionaryOutput:
        target = target if target is not None else x
        x_centered = x - self.b_dec

        total_loss = torch.tensor(0.0, device=x.device)
        loss_dict = {}

        # Compute multi-prefix loss across nested dimensions
        f_full = None
        x_hat_full = None

        for idx, prefix in enumerate(self.prefix_dims):
            w_enc_sub = self.w_enc[:, :prefix]
            b_enc_sub = self.b_enc[:prefix]
            w_dec_sub = self.w_dec[:prefix, :]

            pre_acts = torch.relu(torch.matmul(x_centered, w_enc_sub) + b_enc_sub)
            val, topk_idx = torch.topk(pre_acts, k=min(self.k, prefix), dim=-1)
            f_sub = torch.zeros_like(pre_acts)
            f_sub.scatter_(dim=-1, index=topk_idx, src=val)

            x_hat_sub = torch.matmul(f_sub, w_dec_sub) + self.b_dec
            prefix_mse = nn.functional.mse_loss(x_hat_sub, target)

            weight = 1.0 / len(self.prefix_dims)
            total_loss = total_loss + weight * prefix_mse
            loss_dict[f"mse_prefix_{prefix}"] = prefix_mse

            if prefix == self.d_sae:
                f_full = f_sub
                x_hat_full = x_hat_sub

        if f_full is None:
            f_full = f_sub
            x_hat_full = x_hat_sub

        loss_dict["total_loss"] = total_loss
        return DictionaryOutput(
            reconstructed=x_hat_full,
            feature_acts=f_full,
            loss=total_loss,
            loss_dict=loss_dict,
            extra_dict={"l0": float(self.k), "prefix_dims": self.prefix_dims, "pre_acts": pre_acts}
        )
