from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseTranscoder, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("skip_transcoder")
@register_dictionary("residual_transcoder")
class SkipTranscoder(BaseTranscoder):
    """
    Skip-Transcoder (Residual-Preserving Transcoder).
    y_hat = W_skip x_in + W_dec f(x_in) + b_dec
    Preserves linear residual bypass flow directly via W_skip,
    freeing sparse dictionary capacity to model the non-linear update.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        d_out: Optional[int] = None,
        k: int = 32,
        freeze_skip: bool = False,
        **kwargs
    ):
        d_out = d_out or d_in
        super().__init__(d_in=d_in, d_sae=d_sae, d_out=d_out, k=k, freeze_skip=freeze_skip, **kwargs)
        self.k = k
        self.freeze_skip = freeze_skip

        # Sparse dictionary path
        self.w_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_out))
        self.b_dec = nn.Parameter(torch.zeros(d_out))

        # Linear skip connection bypass
        self.w_skip = nn.Linear(d_in, d_out, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_enc, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        if self.d_in == self.d_out:
            # Initialize skip connection as identity
            nn.init.eye_(self.w_skip.weight)
        else:
            nn.init.kaiming_uniform_(self.w_skip.weight, nonlinearity="linear")
        
        if self.freeze_skip:
            self.w_skip.weight.requires_grad = False
            
        self.normalize_decoder_weights()

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Features encode the non-linear delta over the skip path
        pre_acts = torch.relu(torch.matmul(x, self.w_enc) + self.b_enc)
        val, idx = torch.topk(pre_acts, k=min(self.k, pre_acts.shape[-1]), dim=-1)
        f = torch.zeros_like(pre_acts)
        f.scatter_(dim=-1, index=idx, src=val)
        return f

    def decode(self, f: torch.Tensor, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        sparse_out = torch.matmul(f, self.w_dec) + self.b_dec
        if x is not None:
            return self.w_skip(x) + sparse_out
        return sparse_out

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        dead_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> DictionaryOutput:
        if target is None:
            raise ValueError("SkipTranscoder forward pass requires a 'target' tensor.")

        f = self.encode(x)
        skip_out = self.w_skip(x)
        sparse_out = torch.matmul(f, self.w_dec) + self.b_dec
        y_hat = skip_out + sparse_out

        mse_loss = nn.functional.mse_loss(y_hat, target)

        return DictionaryOutput(
            reconstructed=y_hat,
            feature_acts=f,
            loss=mse_loss,
            loss_dict={"mse_loss": mse_loss, "total_loss": mse_loss},
            extra_dict={"l0": float(self.k), "skip_out": skip_out}
        )
