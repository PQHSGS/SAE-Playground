from typing import Dict
import torch
from src.core.base_dictionary import BaseDictionary


@torch.no_grad()
def compute_feature_splitting_and_absorption(
    dictionary_model: BaseDictionary,
    activations: torch.Tensor,
    max_features_sample: int = 1024,
    absorption_threshold: float = 0.85,
) -> Dict[str, float]:
    """
    Evaluates Feature Splitting, Mutual Coherence, and Feature Absorption
    (Rajamanoharan et al., 2024 / Anthropic Monosemanticity):
    - Mutual Coherence: Max absolute cosine similarity between any two distinct decoder atoms.
    - Mean Atom Similarity: Average cosine similarity between active decoder columns.
    - Jaccard Co-activation Index: Mean Jaccard overlap of active token sets between top pairs.
    - Feature Absorption Score: Fraction of feature pairs where one latent's firing set is
      almost entirely a subset of another broader latent (indicating hierarchical splitting / absorption).
    """
    device = next(dictionary_model.parameters()).device
    activations = activations.to(device)

    # 1. Feature activations
    f = dictionary_model.encode(activations)  # (N, d_sae)
    if f.ndim > 2:
        f = f.view(-1, f.shape[-1])
    num_tokens, d_sae = f.shape

    # 2. Extract unit-normalized decoder weights
    w_dec = dictionary_model.get_decoder_weights()
    if w_dec.ndim == 3:  # Crosscoder: collapse across layers for global coherence
        w_dec = w_dec.mean(dim=0)
    w_norm = w_dec / (torch.norm(w_dec, dim=-1, keepdim=True) + 1e-8)

    # Subsample features if d_sae is very large for fast exact matrix computation
    sample_dim = min(d_sae, max_features_sample)
    active_feature_indices = (f > 0).float().sum(dim=0).topk(k=sample_dim).indices
    w_sample = w_norm[active_feature_indices]  # (sample_dim, d_in)
    f_sample = f[:, active_feature_indices]    # (N, sample_dim)

    # 3. Pairwise Cosine Similarity Matrix
    cosine_matrix = torch.matmul(w_sample, w_sample.T)  # (sample_dim, sample_dim)
    # Mask out diagonal (self-similarity = 1.0)
    diag_mask = torch.eye(sample_dim, device=device, dtype=torch.bool)
    off_diag_cos = cosine_matrix[~diag_mask].abs()

    mutual_coherence = off_diag_cos.max().item() if off_diag_cos.numel() > 0 else 0.0
    mean_coherence = off_diag_cos.mean().item() if off_diag_cos.numel() > 0 else 0.0

    # 4. Binary activation sets and Jaccard Co-activation
    firing_masks = (f_sample > 0).float()  # (N, sample_dim)
    firing_counts = firing_masks.sum(dim=0)  # (sample_dim,)
    
    # Intersections: (sample_dim, sample_dim)
    intersections = torch.matmul(firing_masks.T, firing_masks)
    # Unions: count_i + count_j - intersection_ij
    unions = firing_counts.unsqueeze(0) + firing_counts.unsqueeze(1) - intersections
    jaccard_matrix = intersections / (unions + 1e-8)
    off_diag_jaccard = jaccard_matrix[~diag_mask]
    mean_jaccard = off_diag_jaccard.mean().item() if off_diag_jaccard.numel() > 0 else 0.0

    # 5. Feature Absorption Metric (Rajamanoharan et al., 2024):
    # Subset ratio: P(i active | j active) = Intersection(i, j) / Firing(j)
    subset_ratio = intersections / (firing_counts.unsqueeze(0) + 1e-8)
    # High subset ratio + asymmetric firing count indicates feature absorption
    absorbed_pairs = ((subset_ratio > absorption_threshold) & (firing_counts.unsqueeze(1) > 2 * firing_counts.unsqueeze(0)) & ~diag_mask).float()
    absorption_score = absorbed_pairs.sum().item() / max(1, sample_dim * (sample_dim - 1))

    return {
        "mutual_coherence_max": mutual_coherence,
        "mean_dictionary_coherence": mean_coherence,
        "mean_jaccard_coactivation": mean_jaccard,
        "feature_absorption_score": absorption_score,
    }
