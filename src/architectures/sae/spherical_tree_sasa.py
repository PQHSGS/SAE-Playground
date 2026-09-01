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
    Generalized Dynamic SphericalTreeSASA (Hyperspherical N-Dimensional Subspace Tree SAE).
    
    Combines:
    1. Hyperspherical Cosine-Scored Metric on S^(d-1): Eliminates token norm bias and attention-sink distortion.
    2. Hierarchical 2-Level Tree Topology: Coarse parent concept centroids (M=64) routing to child subspace leaves.
    3. Generalized N-Dimensional Stiefel Subspace Frames: Parameterized as (d_sae, subspace_rank, d_in) with
       strict Gram-Schmidt orthonormalization, eliminating coordinate entanglement.
    4. Exact N-Dimensional Tensorized Reconstruction: Preserves full manifold coordinates without basis starvation.
    5. Subspace Auxiliary Residual Loss (L_sub_aux): Continuous softplus gradient flow into dead subspaces,
       eliminating the winner-takes-all domination cliff in middle layers.
    6. Bounded Geometric Bias: Parameterized via 0.5 * tanh(b_raw) to prevent permanent geometric freeze.
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

        # 2. Level 2: Generalized N-Dimensional Subspace Frames
        # Primary basis w_dec is a leaf parameter for direct DLA attribution and optimizer tracking
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_in))
        if self.subspace_rank > 1:
            self.w_rot = nn.Parameter(torch.empty(d_sae, self.subspace_rank - 1, d_in))
        else:
            self.register_parameter("w_rot", None)
        
        # Bounded encoder bias raw parameter (mapped via 0.5 * tanh)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self._reset_parameters()

    def _reset_parameters(self):
        # Tied transpose unit-sphere initialization for coarse parents
        nn.init.kaiming_uniform_(self.w_coarse_dec, nonlinearity="linear")
        with torch.no_grad():
            self.w_coarse_dec.div_(self.w_coarse_dec.norm(dim=-1, keepdim=True) + 1e-8)
            self.w_coarse_enc.copy_(self.w_coarse_dec.T)

        # Initialize primary and rotation basis vectors with Kaiming uniform
        nn.init.kaiming_uniform_(self.w_dec, nonlinearity="linear")
        if self.subspace_rank > 1 and self.w_rot is not None:
            nn.init.kaiming_uniform_(self.w_rot, nonlinearity="linear")
        self.normalize_decoder_weights()

        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)

    def _get_subspace_basis(self) -> torch.Tensor:
        """
        Returns unified subspace basis tensor of shape (d_sae, subspace_rank, d_in).
        """
        if self.subspace_rank == 1 or self.w_rot is None:
            return self.w_dec.unsqueeze(1)
        return torch.cat([self.w_dec.unsqueeze(1), self.w_rot], dim=1)

    def get_decoder_weights(self) -> torch.Tensor:
        """
        Returns primary decoder basis directions of shape (d_sae, d_in) for attribution / DLA.
        """
        return self.w_dec

    @torch.no_grad()
    def normalize_decoder_weights(self, eps: float = 1e-8, **kwargs) -> None:
        """
        Enforces unit-norm constraints on coarse centroids and in-place Gram-Schmidt
        orthonormalization on the Stiefel manifold for all N-dimensional subspace frames.
        """
        self.w_coarse_dec.div_(self.w_coarse_dec.norm(dim=-1, keepdim=True) + eps)
        
        # 1. Normalize primary decoder basis w_dec
        self.w_dec.div_(self.w_dec.norm(dim=-1, keepdim=True) + eps)
        
        # 2. Orthonormalize rotation bases against primary and preceding bases
        if self.subspace_rank > 1 and self.w_rot is not None:
            for r in range(self.subspace_rank - 1):
                v = self.w_rot[:, r, :].clone()
                # Project out w_dec
                proj_dec = (v * self.w_dec).sum(dim=-1, keepdim=True) * self.w_dec
                v = v - proj_dec
                # Project out preceding rot bases
                for prev_r in range(r):
                    u = self.w_rot[:, prev_r, :]
                    proj_rot = (v * u).sum(dim=-1, keepdim=True) * u
                    v = v - proj_rot
                v.div_(v.norm(dim=-1, keepdim=True) + eps)
                self.w_rot[:, r, :].copy_(v)

    def encode(
        self,
        x: torch.Tensor,
        return_pre_acts: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
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

        # 3. Stage 2: Generalized N-Dimensional Subspace Grassmannian Projection
        w_sub = self._get_subspace_basis()
        # coords: (..., d_sae, subspace_rank)
        coords = torch.einsum('...d, sid -> ...si', x_unit, w_sub)
        
        # Grassmannian subspace energy in [0, 1]
        subspace_energy = torch.norm(coords, p=2, dim=-1)  # (..., d_sae)
        
        # Bounded geometric bias prevents runaway negative drift
        eff_b_enc = 0.5 * torch.tanh(self.b_enc)
        raw_pre_acts = subspace_energy + eff_b_enc
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

        # Modulate full coordinate tensor by sparse firing ratio
        scale_mod = (f_unit / (subspace_energy + 1e-8)).unsqueeze(-1)  # (..., d_sae, 1)
        sparse_coords = coords * scale_mod  # (..., d_sae, subspace_rank)

        # Primary scalar activation for downstream interpretability
        f = f_unit * token_norm

        if return_pre_acts:
            return f, coarse_acts, raw_pre_acts, coords, sparse_coords, token_norm
        return f

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """
        Standard 1D decoder projection interface.
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
        f, coarse_acts, raw_pre_acts, coords, sparse_coords, token_norm = self.encode(x, return_pre_acts=True)
        w_sub = self._get_subspace_basis()
        
        # 1. Exact Lossless N-Dimensional Tensor Reconstruction
        x_hat = token_norm * torch.einsum('...si, sid -> ...d', sparse_coords, w_sub) + self.b_dec
        mse_loss = F.mse_loss(x_hat, target)

        loss_dict = {"mse_loss": mse_loss}
        total_loss = mse_loss

        # 2. Coarse Auxiliary Loss (guides parent centroid discovery)
        coarse_recon = torch.matmul(coarse_acts * token_norm, self.w_coarse_dec) + self.b_dec
        coarse_loss = self.coarse_loss_coeff * F.mse_loss(coarse_recon, target)
        total_loss = total_loss + coarse_loss
        loss_dict["coarse_loss"] = coarse_loss

        # 3. Subspace Orthogonality Loss (regularizer across multi-basis directions)
        if self.subspace_rank > 1 and self.w_rot is not None:
            # Inner products between distinct basis vectors within each subspace
            basis_gram = torch.einsum('sid, sjd -> sij', w_sub, w_sub)  # (d_sae, rank, rank)
            diag_mask = torch.eye(self.subspace_rank, device=x.device, dtype=torch.bool).unsqueeze(0)
            off_diag = basis_gram.masked_select(~diag_mask)
            ortho_loss = self.ortho_loss_coeff * torch.mean(off_diag ** 2)
            total_loss = total_loss + ortho_loss
            loss_dict["ortho_loss"] = ortho_loss

        # 4. Continuous Subspace Auxiliary Residual Loss for Dead Latents (Anti-Death Engine)
        if dead_mask is not None and dead_mask.any() and self.aux_loss_coeff > 0:
            residual = target - x_hat
            dead_indices = torch.where(dead_mask)[0]
            num_dead = len(dead_indices)
            
            if num_dead > 0:
                dead_w = w_sub[dead_indices]  # (num_dead, subspace_rank, d_in)
                # Project residual onto dead subspace frames
                dead_coords = torch.einsum('...d, sid -> ...si', residual / (token_norm + 1e-8), dead_w)
                dead_energy = torch.norm(dead_coords, p=2, dim=-1)  # (..., num_dead)
                
                eff_b_enc = 0.5 * torch.tanh(self.b_enc)
                dead_pre_acts = F.softplus(dead_energy + eff_b_enc[dead_indices])
                
                dead_k = min(self.k, num_dead)
                dead_val, dead_topk_idx = torch.topk(dead_pre_acts, k=dead_k, dim=-1)
                dead_mask_topk = torch.zeros_like(dead_pre_acts)
                dead_mask_topk.scatter_(dim=-1, index=dead_topk_idx, src=dead_val)
                
                dead_scale = (dead_mask_topk / (dead_energy + 1e-8)).unsqueeze(-1)
                dead_sparse_coords = dead_coords * dead_scale
                
                aux_recon = token_norm * torch.einsum('...si, sid -> ...d', dead_sparse_coords, dead_w)
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
