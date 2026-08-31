from typing import Optional
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseSAE, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("sasa")
@register_dictionary("sparsity_adaptive_sae")
class SASA(BaseSAE):
    """
    SASA (Sparsity-Adaptive Sparse Autoencoder).
    Dynamically predicts token-dependent sparsity k(x) based on activation complexity
    instead of enforcing a fixed constant k.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        min_k: int = 8,
        max_k: int = 64,
        target_k: float = 32.0,
        **kwargs
    ):
        super().__init__(d_in=d_in, d_sae=d_sae, min_k=min_k, max_k=max_k, target_k=target_k, **kwargs)
        self.min_k = min_k
        self.max_k = max_k
        self.target_k = target_k

        self.w_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Dynamic sparsity predictor head
        self.k_predictor = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_enc, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        self.normalize_decoder_weights()

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre_acts = torch.relu(torch.matmul(x_centered, self.w_enc) + self.b_enc)

        # Predict dynamic threshold per token
        k_ratio = self.k_predictor(x_centered)  # (..., 1) in [0, 1]
        k_dynamic = self.min_k + (self.max_k - self.min_k) * k_ratio
        
        # Adaptive thresholding: retain features above adaptive quantile
        k_int = int(k_dynamic.mean().item())
        val, idx = torch.topk(pre_acts, k=min(max(k_int, self.min_k), self.max_k), dim=-1)
        f = torch.zeros_like(pre_acts)
        f.scatter_(dim=-1, index=idx, src=val)
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
        x_centered = x - self.b_dec
        pre_acts = torch.relu(torch.matmul(x_centered, self.w_enc) + self.b_enc)

        # Dynamic sparsity budget prediction
        k_ratio = self.k_predictor(x_centered)
        k_dynamic = self.min_k + (self.max_k - self.min_k) * k_ratio
        
        k_int = int(k_dynamic.mean().item())
        val, idx = torch.topk(pre_acts, k=min(max(k_int, self.min_k), self.max_k), dim=-1)
        f = torch.zeros_like(pre_acts)
        f.scatter_(dim=-1, index=idx, src=val)

        x_hat = self.decode(f)
        mse_loss = nn.functional.mse_loss(x_hat, target)

        # Budget loss: enforces mean dynamic k to converge to target_k
        budget_loss = ((k_dynamic.mean() - self.target_k) / max(1.0, self.target_k)) ** 2
        total_loss = mse_loss + 0.05 * budget_loss

        return DictionaryOutput(
            reconstructed=x_hat,
            feature_acts=f,
            loss=total_loss,
            loss_dict={"mse_loss": mse_loss, "budget_loss": budget_loss, "total_loss": total_loss},
            extra_dict={
                "l0": (f > 0).float().sum(dim=-1).mean().item(),
                "mean_dynamic_k": k_dynamic.mean().item(),
                "pre_acts": pre_acts,
            }
        )
