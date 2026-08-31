import os
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from src.core.base_dictionary import BaseDictionary
from src.core.config import TrainingConfig
from src.core.activation_buffer import ActivationBuffer
from src.training.optimizers import ConstrainedAdamW
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
    Production training engine for Sparse Autoencoders, Transcoders, Crosscoders,
    and MultiLayerDictionary ensembles.
    
    Supports:
    1. Single dictionary models (TopK, BatchTopK, JumpReLU, Gated, Crosscoder, etc.)
    2. MultiLayerDictionary ensembles (simultaneous multi-layer training with shared LLM forward pass)
    3. Extreme Low-VRAM optimizations (< 2GB - 4GB GPU memory footprint)
    4. Ghost Gradients, unit-norm constraints, and fault-tolerant per-layer gradient tracking.
    """

    def __init__(
        self,
        dictionary_model: Union[BaseDictionary, nn.Module],
        activation_buffer: ActivationBuffer,
        config: TrainingConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = dictionary_model.to(device)
        self.buffer = activation_buffer
        self.config = config
        self.device = device
        self.console = Console()
        self.total_tokens_trained = 0

        from src.core.multi_dictionary import MultiLayerDictionary
        self.is_multi = isinstance(self.model, MultiLayerDictionary)

        if self.is_multi:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )
            raw_optimizer = self.optimizer
            self.tokens_since_activated: Dict[str, torch.Tensor] = {
                k: torch.zeros(sae.d_sae, dtype=torch.long, device=device)
                for k, sae in self.model.dictionaries.items()
            }
        else:
            self.optimizer = ConstrainedAdamW(
                self.model,
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            raw_optimizer = self.optimizer.optimizer
            self.tokens_since_activated = torch.zeros(
                dictionary_model.d_sae, dtype=torch.long, device=device
            )

        self.scheduler = get_cosine_schedule_with_warmup(
            raw_optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.total_steps,
            min_lr_ratio=config.min_learning_rate / max(1e-8, config.learning_rate),
        )

        self._init_wandb()

    def _init_wandb(self):
        if HAS_WANDB and self.config.wandb_project:
            try:
                run_name = self.config.wandb_run_name or f"{self.model.__class__.__name__}_{int(time.time())}"
                wandb_meta = {
                    "architecture": self.model.__class__.__name__,
                    **self.config.model_dump(),
                }
                if self.is_multi:
                    wandb_meta["num_layers"] = len(self.model.dictionaries)
                    wandb_meta["hook_points"] = list(self.model.hook_point_map.values())
                else:
                    wandb_meta["d_in"] = getattr(self.model, "d_in", None)
                    wandb_meta["d_sae"] = getattr(self.model, "d_sae", None)
                    wandb_meta["d_out"] = getattr(self.model, "d_out", None)

                wandb.init(
                    project=self.config.wandb_project,
                    name=run_name,
                    entity=self.config.wandb_entity,
                    config=wandb_meta,
                    reinit=True,
                )
                logger.info(f"WandB initialized: project='{self.config.wandb_project}', run='{run_name}'")
            except Exception as e:
                logger.warning(f"WandB initialization failed/skipped: {e}")

    def train_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        # Multi-Layer Dictionary Ensemble Step
        if self.is_multi:
            metrics: Dict[str, float] = {}
            total_loss_accum = 0.0
            tokens_in_batch = next(iter(batch.values())).shape[0]
            self.total_tokens_trained += tokens_in_batch

            for hp, act in batch.items():
                sanitized_key = self.model.reverse_map[hp]
                sae = self.model.dictionaries[sanitized_key]
                x_in = act.to(self.device, non_blocking=True).contiguous()

                dead_mask = self.tokens_since_activated[sanitized_key] > self.config.dead_feature_threshold

                try:
                    out = sae(x_in, target=x_in, dead_mask=dead_mask)
                    layer_loss = out.loss

                    if self.config.use_ghost_grads and dead_mask.any() and "pre_acts" in out.extra_dict:
                        residual = (x_in - out.reconstructed).contiguous()
                        ghost_loss = compute_ghost_gradients_loss(
                            residual=residual,
                            pre_acts=out.extra_dict["pre_acts"],
                            w_dec=sae.get_decoder_weights(),
                            dead_mask=dead_mask,
                            ghost_grad_coeff=self.config.ghost_grad_coeff,
                        )
                        layer_loss = layer_loss + ghost_loss

                    layer_loss.backward()
                    total_loss_accum += layer_loss.item()

                    with torch.no_grad():
                        fired = (out.feature_acts > 0).any(dim=list(range(out.feature_acts.ndim - 1)))
                        self.tokens_since_activated[sanitized_key][fired] = 0
                        self.tokens_since_activated[sanitized_key][~fired] += tokens_in_batch

                    nmse = normalized_mse_loss(out.reconstructed, x_in).item()
                    l0 = out.extra_dict.get("l0", (out.feature_acts > 0).float().sum(dim=-1).mean().item())
                    dead_pct = (dead_mask.sum().item() / sae.d_sae) * 100.0

                    metrics[f"layers/{sanitized_key}/nmse"] = nmse
                    metrics[f"layers/{sanitized_key}/l0"] = l0
                    metrics[f"layers/{sanitized_key}/dead_pct"] = dead_pct

                except Exception as e:
                    logger.error(f"Error in training step for layer '{hp}': {e}. Skipping gradient step.")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.model.normalize_decoder_weights()

            num_layers = len(self.model.dictionaries)
            mean_l0 = sum(v for k, v in metrics.items() if k.endswith("/l0")) / max(1, num_layers)
            max_dead_pct = max([v for k, v in metrics.items() if k.endswith("/dead_pct")] or [0.0])

            metrics["loss/total"] = total_loss_accum
            metrics["metrics/mean_l0"] = mean_l0
            metrics["metrics/max_dead_pct"] = max_dead_pct
            metrics["metrics/tokens_processed"] = self.total_tokens_trained
            metrics["metrics/lr"] = self.scheduler.get_last_lr()[0]
            return metrics

        # Single Dictionary Model Step
        if isinstance(batch, (tuple, list)):
            x_in = batch[0].to(self.device, non_blocking=True).contiguous()
            target = batch[1].to(self.device, non_blocking=True).contiguous()
        else:
            x_in = batch.to(self.device, non_blocking=True).contiguous()
            target = x_in

        tokens_in_batch = x_in.shape[0]
        self.total_tokens_trained += tokens_in_batch

        dead_mask = self.tokens_since_activated > self.config.dead_feature_threshold
        out = self.model(x_in, target=target, dead_mask=dead_mask)
        total_loss = out.loss

        ghost_loss_val = 0.0
        if self.config.use_ghost_grads and dead_mask.any() and "pre_acts" in out.extra_dict:
            residual = (target - out.reconstructed).contiguous()
            w_dec = self.model.get_decoder_weights()
            ghost_loss = compute_ghost_gradients_loss(
                residual=residual,
                pre_acts=out.extra_dict["pre_acts"],
                w_dec=w_dec,
                dead_mask=dead_mask,
                ghost_grad_coeff=self.config.ghost_grad_coeff,
            )
            total_loss = total_loss + ghost_loss
            ghost_loss_val = ghost_loss.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        with torch.no_grad():
            fired = (out.feature_acts > 0).any(dim=list(range(out.feature_acts.ndim - 1)))
            self.tokens_since_activated[fired] = 0
            self.tokens_since_activated[~fired] += tokens_in_batch

        nmse = normalized_mse_loss(out.reconstructed, target).item()
        l0 = out.extra_dict.get("l0", (out.feature_acts > 0).float().sum(dim=-1).mean().item())
        dead_count = dead_mask.sum().item()
        dead_pct = (dead_count / self.model.d_sae) * 100.0

        metrics = {
            "loss/total": total_loss.item(),
            "loss/nmse": nmse,
            "loss/ghost_grads": ghost_loss_val,
            "metrics/l0": l0,
            "metrics/dead_count": dead_count,
            "metrics/dead_pct": dead_pct,
            "metrics/tokens_processed": self.total_tokens_trained,
            "metrics/lr": self.scheduler.get_last_lr()[0],
        }

        for k, v in out.loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics[f"loss/{k}"] = v.item()

        return metrics

    def train(self) -> None:
        model_name = self.model.__class__.__name__
        if self.is_multi:
            logger.info(f"Starting Multi-Layer Training across {len(self.model.dictionaries)} layers simultaneously.")
        else:
            logger.info(f"Starting Training: {model_name} (d_in={self.model.d_in}, d_sae={self.model.d_sae})")
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
                f"Training {model_name}...",
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

                l0_val = metrics.get("metrics/l0", metrics.get("metrics/mean_l0", 0.0))
                dead_val = metrics.get("metrics/dead_pct", metrics.get("metrics/max_dead_pct", 0.0))

                progress.update(
                    task,
                    advance=1,
                    loss=running_loss,
                    l0=l0_val,
                    dead=dead_val,
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
