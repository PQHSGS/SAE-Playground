from typing import Iterator, Optional, Tuple
import torch
from torch.optim import AdamW
from src.core.base_dictionary import BaseDictionary


class ConstrainedAdamW:
    """
    AdamW Optimizer with strict unit-norm decoder weight projection after every optimizer step.
    Ensures ||W_dec[i]||_2 = 1.0 continuously, preventing scale drift.
    """

    def __init__(
        self,
        dictionary_model: BaseDictionary,
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.dictionary_model = dictionary_model
        self.optimizer = AdamW(
            dictionary_model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> Optional[float]:
        loss = self.optimizer.step(closure=closure)
        # Enforce unit-norm constraint on decoder weights
        self.dictionary_model.normalize_decoder_weights()
        return loss

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
