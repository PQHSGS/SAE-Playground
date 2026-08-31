from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from src.core.base_dictionary import BaseSAE, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("matching_pursuit")
@register_dictionary("omp_sae")
@register_dictionary("matching_pursuit_sae")
class MatchingPursuitSAE(BaseSAE):
    """
    Matching Pursuit Sparse Autoencoder (Ground-Truth Tied OMP Formulation).
    Uses tied decoder weights W_enc = W_dec^T and unrolls greedy residual projection:
    In each iteration:
      1. Find feature index i = argmax(ReLU(residual @ W_dec^T))
      2. Project residual onto selected decoder row: val = ReLU(residual · W_dec[i])
      3. Update activation: f[i] += val
      4. Update residual: residual -= val * W_dec[i]
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        max_iterations: int = 16,
        residual_threshold: float = 1e-3,
        **kwargs
    ):
        super().__init__(d_in=d_in, d_sae=d_sae, max_iterations=max_iterations, residual_threshold=residual_threshold, **kwargs)
        self.max_iterations = max_iterations
        self.residual_threshold = residual_threshold

        # Tied dictionary: W_dec is the primary parameter, W_enc = W_dec^T
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        self.normalize_decoder_weights()

    def get_decoder_weights(self) -> torch.Tensor:
        return self.w_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = (x - self.b_dec).contiguous()
        original_shape = x_centered.shape

        if x_centered.ndim > 2:
            x_flat = x_centered.view(-1, self.d_in)
        else:
            x_flat = x_centered

        batch_size = x_flat.shape[0]
        residual = x_flat.clone()
        acts = torch.zeros(batch_size, self.d_sae, device=self.w_dec.device, dtype=x.dtype)
        done = torch.zeros(batch_size, dtype=torch.bool, device=self.w_dec.device)

        for _ in range(self.max_iterations):
            # Step 1: Find best matching latent index per token
            with torch.no_grad():
                projections = torch.relu(torch.matmul(residual, self.w_dec.T))
                indices = projections.max(dim=-1, keepdim=True).indices  # (batch_size, 1)
                indices_flat = indices.squeeze(-1)

            # Step 2: Differentiable projection onto selected decoder row
            selected_dec = self.w_dec[indices_flat]  # (batch_size, d_in)
            values = torch.relu((residual * selected_dec).sum(dim=-1, keepdim=True))

            active_mask = (~done).unsqueeze(1)
            masked_values = (values * active_mask.to(values.dtype)).to(acts.dtype)

            # Step 3: Accumulate into activation vector
            acts.scatter_add_(1, indices, masked_values)

            # Step 4: Subtract reconstructed component from residual
            residual = residual - masked_values * selected_dec

            with torch.no_grad():
                res_norm = torch.norm(residual, p=2, dim=-1)
                done = done | (res_norm < self.residual_threshold)
                if done.all():
                    break

        if len(original_shape) > 2:
            acts = acts.view(*original_shape[:-1], self.d_sae)

        return acts

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
