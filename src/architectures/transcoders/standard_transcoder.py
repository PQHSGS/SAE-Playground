from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseTranscoder, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("transcoder")
@register_dictionary("standard_transcoder")
class StandardTranscoder(BaseTranscoder):
    """
    Standard Transcoder (x_in -> sparse features f -> y_out).
    Maps input activations to downstream layer activations or replaces MLP sublayers.
    """

    def __init__(self, d_in: int, d_sae: int, d_out: Optional[int] = None, k: int = 32, **kwargs):
        d_out = d_out or d_in
        super().__init__(d_in=d_in, d_sae=d_sae, d_out=d_out, k=k, **kwargs)
        self.k = k

        self.w_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_out))
        self.b_dec = nn.Parameter(torch.zeros(d_out))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_enc, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        self.normalize_decoder_weights()

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = torch.relu(torch.matmul(x, self.w_enc) + self.b_enc)
        val, idx = torch.topk(pre_acts, k=min(self.k, pre_acts.shape[-1]), dim=-1)
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
        if target is None:
            raise ValueError("Transcoder forward pass requires a 'target' tensor (e.g. y_out).")

        f = self.encode(x)
        y_hat = self.decode(f)
        mse_loss = nn.functional.mse_loss(y_hat, target)

        return DictionaryOutput(
            reconstructed=y_hat,
            feature_acts=f,
            loss=mse_loss,
            loss_dict={"mse_loss": mse_loss, "total_loss": mse_loss},
            extra_dict={"l0": float(self.k)}
        )
