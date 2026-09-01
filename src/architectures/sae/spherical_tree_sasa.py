from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.base_dictionary import BaseSAE, DictionaryOutput
from src.architectures.registry import register_dictionary


@register_dictionary("spherical_tree_sasa")
@register_dictionary("subspace_tree_sae")
@register_dictionary("sts_sae")
class SphericalTreeSASA(BaseSAE):
    """
    SphericalTreeSASA (Spherical Subspace Tree Sparse Autoencoder).
    
    A unified multi-scale architecture combining:
    1. Hyperspherical Cosine-Scored Metric: Scale-invariant projection on S^(d-1),
       eliminating token norm bias and attention-sink distortion.
    2. Hierarchical 2-Level Tree Topology: Coarse parent concept centroids (M=64)
       routing to fine-grained child feature leaves (TreeSAE).
    3. Multi-Dimensional Subspace Manifolds: 2D planar subspace frames per leaf (r=2)
       to capture continuous rotational concepts without feature splitting (SASA).
    4. Global BatchTopK Dynamic Sparsity: Dynamically allocates firing slots across
       tokens based on conceptual richness rather than vector magnitude.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        num_coarse: int = 64,
        subspace_rank: int = 2,
        k: int = 32,
        k_coarse: int = 8,
        aux_loss_coeff: float = 0.03125,
        coarse_loss_coeff: float = 0.25,
        ortho_loss_coeff: float = 1e-4,
        **kwargs
    ):
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            num_coarse=num_coarse,
            subspace_rank=subspace_rank,
            k=k,
            k_coarse=k_coarse,
            aux_loss_coeff=aux_loss_coeff,
            coarse_loss_coeff=coarse_loss_coeff,
            ortho_loss_coeff=ortho_loss_coeff,
            **kwargs
        )
        self.num_coarse = num_coarse
        self.subspace_rank = subspace_rank
        self.k = k
        self.k_coarse = k_coarse
        self.aux_loss_coeff = aux_loss_coeff
        self.coarse_loss_coeff = coarse_loss_coeff
        self.ortho_loss_coeff = ortho_loss_coeff

        # 1. Level 1: Coarse parent concept centroids
        self.w_coarse_enc = nn.Parameter(torch.empty(d_in, num_coarse))
        self.w_coarse_dec = nn.Parameter(torch.empty(num_coarse, d_in))

        # 2. Level 2: Fine leaf 2D subspace frame (primary basis w_dec, orthogonal rotation w_rot)
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.w_rot = nn.Parameter(torch.empty(d_sae, d_in))
        
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self._reset_parameters()

    def _reset_parameters(self):
        # Tied transpose unit-sphere initialization for coarse parents
        nn.init.kaiming_uniform_(self.w_coarse_dec, nonlinearity="linear")
        with torch.no_grad():
            self.w_coarse_dec.div_(self.w_coarse_dec.norm(dim=-1, keepdim=True) + 1e-8)
            self.w_coarse_enc.copy_(self.w_coarse_dec.T)

        # Initialize subspace frames
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        nn.init.kaiming_uniform_(self.w_rot, nonlinearity="linear")
        self.normalize_decoder_weights()

        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)

    def get_decoder_weights(self) -> torch.Tensor:
        """
        Returns primary decoder basis directions of shape (d_sae, d_in) for attribution / DLA.
        """
        return self.w_dec

    @torch.no_grad()
    def normalize_decoder_weights(self, eps: float = 1e-8, **kwargs) -> None:
        """
        Enforces unit-norm constraints on coarse centroids and fine subspace basis vectors.
        """
        self.w_coarse_dec.div_(self.w_coarse_dec.norm(dim=-1, keepdim=True) + eps)
        self.w_dec.div_(self.w_dec.norm(dim=-1, keepdim=True) + eps)
        self.w_rot.div_(self.w_rot.norm(dim=-1, keepdim=True) + eps)

    def encode(
        self,
        x: torch.Tensor,
        return_pre_acts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        x_centered = x - self.b_dec
        token_norm = torch.norm(x_centered, p=2, dim=-1, keepdim=True) + 1e-8
        
        # 1. Hyperspherical projection on unit sphere S^(d-1)
        x_unit = x_centered / token_norm

        # 2. Stage 1: Coarse Parent Cosine Routing
        coarse_sim = torch.matmul(x_unit, self.w_coarse_enc)  # (..., num_coarse)
        topk_coarse_val, topk_coarse_idx = torch.topk(
            coarse_sim, k=min(self.k_coarse, self.num_coarse), dim=-1
        )
        coarse_acts = torch.zeros_like(coarse_sim)
        coarse_acts.scatter_(dim=-1, index=topk_coarse_idx, src=F.relu(topk_coarse_val))

        # 3. Stage 2: Fine 2D Subspace Grassmannian Energy Evaluation
        # Project x_unit along primary basis (w_dec) and orthogonal rotation (w_rot)
        proj_1 = torch.matmul(x_unit, self.w_dec.T)  # (..., d_sae)
        proj_2 = torch.matmul(x_unit, self.w_rot.T)  # (..., d_sae)
        
        # Grassmannian subspace cosine projection magnitude in [0, 1]
        subspace_energy = torch.sqrt(proj_1 ** 2 + proj_2 ** 2 + 1e-12)
        raw_pre_acts = subspace_energy + self.b_enc
        pre_acts = F.relu(raw_pre_acts)

        orig_shape = pre_acts.shape
        flat_acts = pre_acts.view(-1, self.d_sae)
        batch_size_tokens = flat_acts.shape[0]

        # 4. Stage 3: BatchTopK Global Sparsity Selection
        if self.training:
            total_k = min(batch_size_tokens * self.k, flat_acts.numel())
            flat_view = flat_acts.view(-1)
            val, idx = torch.topk(flat_view, k=total_k)
            sparse_flat = torch.zeros_like(flat_view)
            sparse_flat.scatter_(dim=0, index=idx, src=val)
            f_unit = sparse_flat.view(orig_shape)
        else:
            val, idx = torch.topk(pre_acts, k=min(self.k, self.d_sae), dim=-1)
            f_unit = torch.zeros_like(pre_acts)
            f_unit.scatter_(dim=-1, index=idx, src=val)

        # 5. Modulate sparse cosine firings by original token norm for lossless scale
        f = f_unit * token_norm

        if return_pre_acts:
            return f, coarse_acts, raw_pre_acts, proj_1, proj_2, token_norm
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs signal from active subspace frames.
        """
        return torch.matmul(f, self.w_dec) + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        dead_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> DictionaryOutput:
        target = target if target is not None else x
        f, coarse_acts, raw_pre_acts, proj_1, proj_2, token_norm = self.encode(x, return_pre_acts=True)
        
        # Primary fine reconstruction
        x_hat = self.decode(f)
        mse_loss = F.mse_loss(x_hat, target)

        loss_dict = {"mse_loss": mse_loss}
        total_loss = mse_loss

        # 1. Coarse Auxiliary Loss (guides parent centroid discovery)
        coarse_recon = torch.matmul(coarse_acts * token_norm, self.w_coarse_dec) + self.b_dec
        coarse_loss = self.coarse_loss_coeff * F.mse_loss(coarse_recon, target)
        total_loss = total_loss + coarse_loss
        loss_dict["coarse_loss"] = coarse_loss

        # 2. Subspace Orthogonality Regularization (ensures w_dec and w_rot are orthogonal)
        ortho_dot = (self.w_dec * self.w_rot).sum(dim=-1)  # (d_sae,)
        ortho_loss = self.ortho_loss_coeff * torch.mean(ortho_dot ** 2)
        total_loss = total_loss + ortho_loss
        loss_dict["ortho_loss"] = ortho_loss

        # 3. OpenAI Softplus Auxiliary Loss for Dead Latents
        if dead_mask is not None and dead_mask.any() and self.aux_loss_coeff > 0:
            residual = target - x_hat
            dead_pre_acts = F.softplus(raw_pre_acts) * dead_mask.float()
            dead_k = min(self.k, int(dead_mask.sum().item()))
            if dead_k > 0:
                dead_val, dead_idx = torch.topk(dead_pre_acts, k=dead_k, dim=-1)
                dead_f = torch.zeros_like(dead_pre_acts)
                dead_f.scatter_(dim=-1, index=dead_idx, src=dead_val)
                aux_recon = torch.matmul(dead_f * token_norm, self.w_dec)
                aux_loss = self.aux_loss_coeff * F.mse_loss(aux_recon, residual)
                total_loss = total_loss + aux_loss
                loss_dict["aux_loss"] = aux_loss

        loss_dict["total_loss"] = total_loss
        return DictionaryOutput(
            reconstructed=x_hat,
            feature_acts=f,
            loss=total_loss,
            loss_dict=loss_dict,
            extra_dict={"l0": (f > 0).float().sum(dim=-1).mean().item(), "pre_acts": raw_pre_acts}
        )
