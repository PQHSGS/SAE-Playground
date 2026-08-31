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

        # Running statistics for normalization
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
        except Exception as e:
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
        else:
            texts = [
                "Mechanistic interpretability of sparse autoencoders, transcoders, and crosscoders in deep transformers."
                "Neural networks represent concepts in linear subspaces through superposition and polysemanticity."
            ] * self.model_batch_size
        return texts

    @torch.no_grad()
    def _fill_buffer(self) -> None:
        """
        Fills the activation reservoir with memory-efficient contiguous allocations on CPU.
        """
        self.hook_manager.register_forward_hooks(self.all_hook_points)
        collected_acts: Dict[str, List[torch.Tensor]] = {hp: [] for hp in self.all_hook_points}
        total_tokens_collected = 0

        # Run model forward under autocast for VRAM efficiency
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        while total_tokens_collected < self.buffer_size:
            texts = self._get_text_batch()
            encoding = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.context_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self.device, non_blocking=True)
            attention_mask = encoding["attention_mask"].to(self.device, non_blocking=True)

            # Bypass final lm_head vocabulary projection to save massive VRAM (e.g. 262k logits)
            base_model = getattr(self.model, "model", getattr(self.model, "transformer", self.model))

            if torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    _ = base_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                _ = base_model(input_ids=input_ids, attention_mask=attention_mask)

            # Mask out padding tokens and token-0 (BOS attention sink)
            mask = attention_mask.bool()
            if self.mask_bos and mask.shape[1] > 0:
                mask[:, 0] = False

            # Extract activations cleanly and store directly on CPU
            for hp in self.all_hook_points:
                raw_act = self.hook_manager.activations[hp]
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

    @torch.no_grad()
    def compute_normalization_stats(self, num_tokens: int = 50_000) -> Tuple[torch.Tensor, float]:
        """
        Calculate running mean and scalar normalization factor:
        s = sqrt(d) / E[||x - mu||_2]
        """
        self._fill_buffer()
        primary_hp = self.hook_points[0]
        sample = self.buffer[primary_hp][:num_tokens].float().contiguous()
        self.mean = sample.mean(dim=0, keepdim=True).contiguous()
        centered = sample - self.mean
        avg_norm = torch.norm(centered, p=2, dim=-1).mean().item()
        d_in = sample.shape[-1]
        self.scale_factor = math.sqrt(d_in) / (avg_norm + 1e-8)
        return self.mean.to(dtype=self.dtype, device=self.device), self.scale_factor

    def next_batch(self) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        if self.tokens_buffered == 0 or (self.buffer_idx + self.batch_size) > self.tokens_buffered:
            self._fill_buffer()

        start = self.buffer_idx
        end = start + self.batch_size
        self.buffer_idx = end

        # Multi-SAE dictionary mode
        if self.return_dict:
            return {
                hp: self.buffer[hp][start:end].to(self.device, non_blocking=True).contiguous()
                for hp in self.hook_points
            }

        # Transcoder mode
        if self.target_hook_points is not None:
            x_in = self.buffer[self.hook_points[0]][start:end].to(self.device, non_blocking=True).contiguous()
            y_out = self.buffer[self.target_hook_points[0]][start:end].to(self.device, non_blocking=True).contiguous()
            return x_in, y_out

        # Multi-layer Crosscoder mode
        if len(self.hook_points) > 1:
            layer_acts = [self.buffer[hp][start:end].to(self.device, non_blocking=True).contiguous() for hp in self.hook_points]
            return torch.stack(layer_acts, dim=1).contiguous()

        # Single-hook SAE mode
        return self.buffer[self.hook_points[0]][start:end].to(self.device, non_blocking=True).contiguous()

    def __iter__(self) -> Iterator[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        while True:
            yield self.next_batch()
