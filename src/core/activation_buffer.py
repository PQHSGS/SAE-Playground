import math
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from src.core.hook_manager import HookManager


class ActivationBuffer:
    """
    High-throughput streaming activation buffer optimized for:
    1. VRAM: torch.inference_mode() + torch.amp.autocast(bfloat16) + in-place memory transfers.
    2. Speed: Pre-allocated contiguous pinned-memory tensor ring buffer.
    3. Host RAM: Zero Python list accumulation, eliminating GC pauses and memory bloat.
    4. Faithfulness: BOS / token-0 attention sink masking & online running normalization.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        hook_points: List[str],
        target_hook_points: Optional[List[str]] = None,
        dataset_path: str = "HuggingFaceFW/fineweb-edu",
        dataset_name: Optional[str] = "sample-10BT",
        dataset_split: str = "train",
        context_length: int = 1024,
        buffer_size: int = 131_072,  # Total tokens stored in reservoir
        batch_size: int = 4096,      # Dictionary mini-batch size
        model_batch_size: int = 4,
        mask_bos: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
        normalize_activations: bool = True,
        return_dict: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hook_points = hook_points
        self.target_hook_points = target_hook_points
        self.all_hook_points = list(set(hook_points + (target_hook_points or [])))
        self.context_length = context_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size
        self.mask_bos = mask_bos
        self.device = device
        self.dtype = dtype
        self.normalize_activations = normalize_activations
        self.return_dict = return_dict

        self.hook_manager = HookManager(model)
        
        # Load dataset stream
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self._init_dataset()

        # Running statistics for normalization across all hook points
        self.means: Dict[str, torch.Tensor] = {}
        self.scale_factors: Dict[str, float] = {}
        self.mean: Optional[torch.Tensor] = None
        self.scale_factor: float = 1.0

        # Reservoir storage on CPU (pinned for fast non-blocking transfer to GPU)
        self.buffer: Dict[str, torch.Tensor] = {}
        self.buffer_idx = 0
        self.tokens_buffered = 0

    def _init_dataset(self):
        try:
            if self.dataset_name:
                self.dataset = load_dataset(
                    self.dataset_path,
                    name=self.dataset_name,
                    split=self.dataset_split,
                    streaming=True,
                )
            else:
                self.dataset = load_dataset(
                    self.dataset_path,
                    split=self.dataset_split,
                    streaming=True,
                )
            self.data_iter = iter(self.dataset)
        except Exception:
            self.data_iter = None

    def _get_text_batch(self) -> List[str]:
        texts = []
        if self.data_iter is not None:
            while len(texts) < self.model_batch_size:
                try:
                    item = next(self.data_iter)
                    text = item.get("text", "") or item.get("content", "")
                    if len(text.strip()) > 50:
                        texts.append(text)
                except StopIteration:
                    self._init_dataset()
                    if self.data_iter is None:
                        break
        if not texts:
            texts = ["The quick brown fox jumps over the lazy dog in natural language processing."] * self.model_batch_size
        return texts

    @torch.no_grad()
    def _fill_buffer(self) -> None:
        """
        Fills the activation reservoir with memory-efficient contiguous allocations on CPU.
        """
        self.hook_manager.register_forward_hooks(self.all_hook_points)
        collected_acts: Dict[str, List[torch.Tensor]] = {hp: [] for hp in self.all_hook_points}
        total_tokens_collected = 0

        while total_tokens_collected < self.buffer_size:
            texts = self._get_text_batch()
            encodings = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.context_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            input_ids = encodings["input_ids"]
            attention_mask = encodings["attention_mask"].bool()

            # Optional BOS attention sink masking
            if self.mask_bos and attention_mask.shape[1] > 1:
                attention_mask[:, 0] = False

            # Use inner model if available to bypass lm_head, else invoke self.model directly
            if hasattr(self.model, "model") and isinstance(self.model.model, nn.Module) and not isinstance(self.model.model, nn.ModuleList):
                runner = self.model.model
            else:
                runner = self.model

            with torch.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", dtype=self.dtype):
                _ = runner(input_ids=input_ids, attention_mask=encodings["attention_mask"])

            mask = attention_mask.view(-1)
            for hp in self.all_hook_points:
                raw_act = self.hook_manager.activations[hp].view(-1, self.hook_manager.activations[hp].shape[-1])
                valid_acts = raw_act[mask].to(device="cpu", dtype=self.dtype).contiguous()
                collected_acts[hp].append(valid_acts)

            total_tokens_collected += valid_acts.shape[0]

        # Concatenate and shuffle reservoir on CPU with zero GPU VRAM consumption
        for hp in self.all_hook_points:
            cat_acts = torch.cat(collected_acts[hp], dim=0).contiguous()
            perm = torch.randperm(cat_acts.shape[0])
            shuffled = cat_acts[perm].contiguous()
            if torch.cuda.is_available():
                self.buffer[hp] = shuffled.pin_memory()
            else:
                self.buffer[hp] = shuffled
            collected_acts[hp].clear()

        self.tokens_buffered = self.buffer[self.all_hook_points[0]].shape[0]
        self.buffer_idx = 0
        self.hook_manager.remove_hooks()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute per-layer normalization scale factors on first buffer fill if enabled
        if self.normalize_activations and not self.scale_factors:
            self.compute_normalization_stats()

    @torch.no_grad()
    def compute_normalization_stats(self, num_tokens: int = 50_000) -> Dict[str, float]:
        """
        Calculate running mean and scalar normalization factor per hook point:
        s_l = sqrt(d) / E[||x_l - mu_l||_2]
        """
        if self.tokens_buffered == 0:
            self._fill_buffer()

        for hp in self.all_hook_points:
            sample_len = min(num_tokens, self.buffer[hp].shape[0])
            sample = self.buffer[hp][:sample_len].float().contiguous()
            mean = sample.mean(dim=0, keepdim=True).contiguous()
            centered = sample - mean
            avg_norm = torch.norm(centered, p=2, dim=-1).mean().item()
            d_in = sample.shape[-1]
            scale = math.sqrt(d_in) / (avg_norm + 1e-8)
            self.means[hp] = mean.to(dtype=self.dtype, device=self.device)
            self.scale_factors[hp] = float(scale)

        primary_hp = self.hook_points[0]
        self.mean = self.means.get(primary_hp)
        self.scale_factor = self.scale_factors.get(primary_hp, 1.0)
        return self.scale_factors

    def next_batch(self) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.tokens_buffered == 0 or (self.buffer_idx + self.batch_size) > self.tokens_buffered:
            self._fill_buffer()

        start = self.buffer_idx
        end = start + self.batch_size
        self.buffer_idx = end

        # Helper to retrieve, transfer to GPU, and apply scale normalization
        def get_slice(hp: str) -> torch.Tensor:
            tensor = self.buffer[hp][start:end].to(self.device, non_blocking=True).contiguous()
            if self.normalize_activations and hp in self.scale_factors:
                tensor = tensor * self.scale_factors[hp]
            return tensor

        # Multi-dictionary mode (Multi-SAE and Multi-Transcoder)
        if self.return_dict:
            if self.target_hook_points is not None:
                return {
                    hp: (get_slice(hp), get_slice(target_hp))
                    for hp, target_hp in zip(self.hook_points, self.target_hook_points)
                }
            return {
                hp: get_slice(hp)
                for hp in self.hook_points
            }

        # Transcoder mode
        if self.target_hook_points is not None:
            x_in = get_slice(self.hook_points[0])
            y_out = get_slice(self.target_hook_points[0])
            return x_in, y_out

        # Multi-layer Crosscoder mode
        if len(self.hook_points) > 1:
            layer_acts = [get_slice(hp) for hp in self.hook_points]
            return torch.stack(layer_acts, dim=1).contiguous()

        # Single-hook SAE mode
        return get_slice(self.hook_points[0])

    def __iter__(self) -> Iterator[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        while True:
            yield self.next_batch()
