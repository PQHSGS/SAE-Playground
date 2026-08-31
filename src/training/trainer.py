import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from src.core.base_dictionary import BaseDictionary
from src.core.multi_dictionary import MultiLayerDictionary
from src.core.config import TrainingConfig
from src.core.activation_buffer import ActivationBuffer
from src.training.schedulers import get_cosine_schedule_with_warmup
from src.training.loss_functions import compute_ghost_gradients_loss, normalized_mse_loss
from src.utils.logging import setup_logger

logger = setup_logger("trainer")

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class DictionaryTrainer:
    """
    Unified training engine for Sparse Autoencoders, Transcoders, Crosscoders,
    and Multi-Layer Dictionary ensembles.
    
    Design Principle:
    - Pure Collection Paradigm: All models are treated as a collection of dictionary modules (N >= 1).
    - Single-layer training is simply the N=1 special case of a collection.
    - Zero branching: One unified forward/backward loop handles single-layer, multi-layer, and crosscoders.
    - VRAM Shielded: Sequential per-layer backward pass frees autograd graphs immediately.
    """

    def __init__(
        self,
        dictionary_model: Union[BaseDictionary, MultiLayerDictionary],
        activation_buffer: Optional[ActivationBuffer],
        config: TrainingConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # Always wrap into MultiLayerDictionary collection (N >= 1)
        if isinstance(dictionary_model, BaseDictionary):
            self.model = MultiLayerDictionary({"default": dictionary_model}).to(device)
        elif isinstance(dictionary_model, MultiLayerDictionary):
            self.model = dictionary_model.to(device)
        else:
            raise TypeError(f"Unsupported model type: {type(dictionary_model)}")

        self.buffer = activation_buffer
        self.config = config
        self.device = device
        self.console = Console()
        self.total_tokens_trained = 0

        # Optimizer for all dictionary parameters
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.total_steps,
            min_lr_ratio=config.min_learning_rate / max(1e-8, config.learning_rate),
        )

        # Per-layer dead feature tracking
        self.tokens_since_activated: Dict[str, torch.Tensor] = {
            k: torch.zeros(sae.d_sae, dtype=torch.long, device=device)
            for k, sae in self.model.dictionaries.items()
        }

        self._init_wandb()

    def _init_wandb(self):
        if HAS_WANDB and self.config.wandb_project:
            try:
                first_sae = next(iter(self.model.dictionaries.values()))
                arch_name = first_sae.__class__.__name__
                run_name = self.config.wandb_run_name or f"{arch_name}_{len(self.model)}layers_{int(time.time())}"
                
                wandb.init(
                    project=self.config.wandb_project,
                    name=run_name,
                    entity=self.config.wandb_entity,
                    config={
                        "architecture": arch_name,
                        "num_layers": len(self.model),
                        "hook_points": list(self.model.hook_point_map.values()),
                        "d_in": first_sae.d_in,
                        "d_sae": first_sae.d_sae,
                        **self.config.model_dump(),
                    },
                    reinit=True,
                )
                logger.info(f"WandB initialized: project='{self.config.wandb_project}', run='{run_name}'")
            except Exception as e:
                logger.warning(f"WandB initialization skipped: {e}")

    def train_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        # Normalize any batch shape into standard Dict[str, Tensor] collection
        if isinstance(batch, (torch.Tensor, tuple, list)):
            primary_key = next(iter(self.model.hook_point_map.values()))
            batch_dict = {primary_key: batch}
        else:
            batch_dict = batch

        metrics: Dict[str, float] = {}
        total_loss_accum = 0.0
        
        first_val = next(iter(batch_dict.values()))
        tokens_in_batch = first_val[0].shape[0] if isinstance(first_val, (tuple, list)) else first_val.shape[0]
        self.total_tokens_trained += tokens_in_batch

        # Unified single loop for all N >= 1 dictionary modules
        for hp, act in batch_dict.items():
            sanitized_key = self.model.reverse_map.get(hp, self.model._sanitize_key(hp))
            if sanitized_key not in self.model.dictionaries:
                continue

            sae = self.model.dictionaries[sanitized_key]

            # Transcoder vs standard SAE parsing
            if isinstance(act, (tuple, list)):
                x_in = act[0].to(self.device, non_blocking=True).contiguous()
                target = act[1].to(self.device, non_blocking=True).contiguous()
            else:
                x_in = act.to(self.device, non_blocking=True).contiguous()
                target = x_in

            dead_mask = self.tokens_since_activated[sanitized_key] > self.config.dead_feature_threshold

            try:
                out = sae(x_in, target=target, dead_mask=dead_mask)
                layer_loss = out.loss

                # Ghost gradients for reviving dead latents
                if self.config.use_ghost_grads and dead_mask.any() and "pre_acts" in out.extra_dict:
                    residual = (target - out.reconstructed).contiguous()
                    ghost_loss = compute_ghost_gradients_loss(
                        residual=residual,
                        pre_acts=out.extra_dict["pre_acts"],
                        w_dec=sae.get_decoder_weights(),
                        dead_mask=dead_mask,
                        ghost_grad_coeff=self.config.ghost_grad_coeff,
                    )
                    layer_loss = layer_loss + ghost_loss

                # Immediate backward pass frees autograd graph for x_in (VRAM Shield)
                layer_loss.backward()
                total_loss_accum += layer_loss.item()

                # Dead feature tracker update
                with torch.no_grad():
                    fired = (out.feature_acts > 0).any(dim=list(range(out.feature_acts.ndim - 1)))
                    self.tokens_since_activated[sanitized_key][fired] = 0
                    self.tokens_since_activated[sanitized_key][~fired] += tokens_in_batch

                # Record per-layer metrics
                nmse = normalized_mse_loss(out.reconstructed, target).item()
                l0 = out.extra_dict.get("l0", (out.feature_acts > 0).float().sum(dim=-1).mean().item())
                dead_pct = (dead_mask.sum().item() / sae.d_sae) * 100.0

                prefix = f"layers/{sanitized_key}" if len(self.model) > 1 else "loss"
                metrics[f"{prefix}/nmse"] = nmse
                metrics[f"{prefix}/l0"] = l0
                metrics[f"{prefix}/dead_pct"] = dead_pct

            except Exception as e:
                logger.error(f"Error in training step for '{hp}': {e}. Skipping gradient step.")

        # Clip gradients, step optimizer & scheduler
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Enforce unit-norm decoder weights constraint
        self.model.normalize_decoder_weights()

        # Compute aggregate metrics
        num_layers = len(self.model)
        l0_vals = [v for k, v in metrics.items() if k.endswith("/l0")]
        dead_vals = [v for k, v in metrics.items() if k.endswith("/dead_pct")]

        metrics["loss/total"] = total_loss_accum
        metrics["metrics/mean_l0"] = sum(l0_vals) / max(1, len(l0_vals)) if l0_vals else 0.0
        metrics["metrics/max_dead_pct"] = max(dead_vals) if dead_vals else 0.0
        metrics["metrics/tokens_processed"] = self.total_tokens_trained
        metrics["metrics/lr"] = self.scheduler.get_last_lr()[0]

        return metrics

    def train(self) -> None:
        first_sae = next(iter(self.model.dictionaries.values()))
        arch_name = first_sae.__class__.__name__
        num_layers = len(self.model)

        logger.info(f"Starting Training: {arch_name} across {num_layers} layer(s) (d_in={first_sae.d_in}, d_sae={first_sae.d_sae})")
        logger.info(f"Target steps: {self.config.total_steps} | Batch size: {self.config.batch_size}")

        step = 0
        running_loss = 0.0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("Loss: {task.fields[loss]:.4f} | L0: {task.fields[l0]:.1f} | Dead: {task.fields[dead]:.1f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                f"Training {arch_name}...",
                total=self.config.total_steps,
                loss=0.0,
                l0=0.0,
                dead=0.0,
            )

            for batch in self.buffer:
                step += 1
                metrics = self.train_step(batch)

                running_loss = 0.9 * running_loss + 0.1 * metrics["loss/total"] if step > 1 else metrics["loss/total"]

                if HAS_WANDB and wandb.run is not None:
                    wandb.log(metrics, step=step)

                progress.update(
                    task,
                    advance=1,
                    loss=running_loss,
                    l0=metrics.get("metrics/mean_l0", 0.0),
                    dead=metrics.get("metrics/max_dead_pct", 0.0),
                )

                if step % self.config.checkpoint_steps == 0 or step == self.config.total_steps:
                    save_path = os.path.join(self.config.output_dir, f"step_{step}")
                    self.model.save_pretrained(
                        save_path,
                        metadata={
                            "step": step,
                            "tokens_processed": self.total_tokens_trained,
                            "metrics": metrics,
                        }
                    )
                    logger.info(f"Saved checkpoint to: {save_path} (step={step})")

                if step >= self.config.total_steps:
                    break

        logger.info("Training completed successfully!")
        if HAS_WANDB and wandb.run is not None:
            wandb.finish()


# Backward-compatible alias
MultiSAETrainer = DictionaryTrainer
