from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseSAE, DictionaryOutput
from src.architectures.registry import register_dictionary


class JumpReLUFunction(torch.autograd.Function):
    """
    JumpReLU activation function with Gaussian-kernel Straight-Through Estimator (STE).
    f(x, theta) = x * H(x - theta)
    d/d_theta = - theta * (1/bandwidth) * N((x - theta)/bandwidth)
    """

    @staticmethod
    def forward(ctx, pre_acts: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(pre_acts, threshold)
        ctx.bandwidth = bandwidth
        
        # Step function: 1 if pre_acts > threshold else 0
        mask = (pre_acts > threshold).float()
        return pre_acts * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        pre_acts, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        # Standard gradient for pre_acts (passed through where active)
        mask = (pre_acts > threshold).float()
        grad_pre_acts = grad_output * mask

        # STE gradient for threshold: estimate jump derivative via Gaussian kernel
        diff = (pre_acts - threshold) / (bandwidth + 1e-8)
        gaussian = torch.exp(-0.5 * diff ** 2) / (math.sqrt(2 * math.pi) * (bandwidth + 1e-8))
        
        # Derivative with respect to threshold: - pre_acts * gaussian * grad_output
        grad_threshold = - (grad_output * pre_acts * gaussian).sum(dim=list(range(pre_acts.ndim - 1)))

        return grad_pre_acts, grad_threshold, None


@register_dictionary("jumprelu")
@register_dictionary("jumprelu_sae")
class JumpReLUSAE(BaseSAE):
    """
    JumpReLU Sparse Autoencoder (Rajamanoharan et al., Google DeepMind 2024).
    Uses a learnable per-feature threshold with zero shrinkage for active features.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        l1_coeff: float = 1e-3,
        init_threshold: float = 0.001,
        bandwidth: float = 0.001,
        **kwargs
    ):
        super().__init__(d_in=d_in, d_sae=d_sae, l1_coeff=l1_coeff, **kwargs)
        self.l1_coeff = l1_coeff
        self.bandwidth = bandwidth

        self.w_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        
        # Log-threshold for positivity: theta = exp(log_threshold)
        self.log_threshold = nn.Parameter(torch.full((d_sae,), math.log(init_threshold)))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_enc, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        self.normalize_decoder_weights()

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    @property
    def threshold(self) -> torch.Tensor:
        return torch.exp(self.log_threshold)

    def encode(self, x: torch.Tensor, return_pre_acts: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_centered = x - self.b_dec
        pre_acts = torch.relu(torch.matmul(x_centered, self.w_enc) + self.b_enc)
        f = JumpReLUFunction.apply(pre_acts, self.threshold, self.bandwidth)
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

        mse_loss = nn.functional.mse_loss(x_hat, target)
        l0_proxy = (f > 0).float().sum(dim=-1).mean()
        sparsity_loss = self.l1_coeff * l0_proxy
        total_loss = mse_loss + sparsity_loss

        return DictionaryOutput(
            reconstructed=x_hat,
            feature_acts=f,
            loss=total_loss,
            loss_dict={"mse_loss": mse_loss, "sparsity_loss": sparsity_loss, "total_loss": total_loss},
            extra_dict={
                "l0": l0_proxy.item(),
                "mean_threshold": self.threshold.mean().item(),
                "pre_acts": pre_acts,
            }
        )
