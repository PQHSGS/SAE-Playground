from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from transformers import PreTrainedTokenizerBase
from src.core.base_dictionary import BaseDictionary


@dataclass
class TokenActivation:
    token_str: str
    token_id: int
    activation_val: float


@dataclass
class ActivatingSnippet:
    context_text: str
    tokens: List[TokenActivation]
    max_activation: float
    max_token_idx: int


class FeatureSampleCollector:
    """
    Collects top max-activating and quantile-binned dataset context snippets
    with exact token-level activation values for automated interpretation and UI rendering.
    """

    def __init__(
        self,
        dictionary_model: BaseDictionary,
        tokenizer: PreTrainedTokenizerBase,
        max_samples_per_feature: int = 20,
        quantile_samples: int = 10,
        window_size: int = 32,
        min_activation: float = 0.3,
    ):
        self.dictionary_model = dictionary_model
        self.tokenizer = tokenizer
        self.max_samples = max_samples_per_feature
        self.quantile_samples = quantile_samples
        self.window_size = window_size
        self.min_activation = min_activation

        # Storage: feature_idx -> list of ActivatingSnippet
        self.feature_records: Dict[int, List[ActivatingSnippet]] = {}

    @torch.no_grad()
    def process_sequence_activations(
        self,
        input_ids: torch.Tensor,       # (seq_len,)
        activations: torch.Tensor,     # (seq_len, d_in)
        target_features: Optional[List[int]] = None,
    ) -> None:
        """
        Processes a single tokenized sequence and updates top-activating snippet records.
        """
        scale = (self.dictionary_model.d_in ** 0.5) / (activations.norm(dim=-1, keepdim=True) + 1e-8)
        normed_act = (activations * scale).to(device=self.dictionary_model.get_decoder_weights().device, dtype=self.dictionary_model.get_decoder_weights().dtype)
        with torch.no_grad():
            f = self.dictionary_model.encode(normed_act)  # (seq_len, d_sae)

        seq_len = input_ids.shape[0]
        max_vals, max_positions = f.max(dim=0)  # (d_sae,)

        active_mask = max_vals > self.min_activation
        if target_features:
            active_indices = [idx for idx in target_features if active_mask[idx].item()]
        else:
            active_indices = torch.where(active_mask)[0].tolist()

        for feat_idx in active_indices:
            max_val = float(max_vals[feat_idx].item())
            max_pos = int(max_positions[feat_idx].item())
            feat_acts = f[:, feat_idx]

            # Window bounds around peak activation
            start_pos = max(0, max_pos - self.window_size // 2)
            end_pos = min(seq_len, max_pos + self.window_size // 2)

            snippet_tokens = []
            for i in range(start_pos, end_pos):
                tid = input_ids[i].item()
                t_str = self.tokenizer.decode([tid])
                act_val = feat_acts[i].item()
                snippet_tokens.append(TokenActivation(token_str=t_str, token_id=tid, activation_val=act_val))

            full_text = self.tokenizer.decode(input_ids[start_pos:end_pos])
            record = ActivatingSnippet(
                context_text=full_text,
                tokens=snippet_tokens,
                max_activation=max_val,
                max_token_idx=max_pos - start_pos,
            )

            if feat_idx not in self.feature_records:
                self.feature_records[feat_idx] = []

            records = self.feature_records[feat_idx]
            records.append(record)
            # Keep sorted by max activation descending
            records.sort(key=lambda r: r.max_activation, reverse=True)
            if len(records) > self.max_samples + self.quantile_samples:
                self.feature_records[feat_idx] = records[: self.max_samples + self.quantile_samples]

    def get_feature_snippets(self, feat_idx: int) -> List[ActivatingSnippet]:
        return self.feature_records.get(feat_idx, [])
