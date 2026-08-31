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
    Production training engine for Sparse Autoencoders, Transcoders, and Crosscoders.
    Supports Ghost Gradients, unit-norm constraints, mixed precision, and structured tracking.
    """

    def __init__(
        self,
        dictionary_model: BaseDictionary,
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

        # Optimizer with decoder normalization constraint
        self.optimizer = ConstrainedAdamW(
            self.model,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Learning rate scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer.optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.total_steps,
            min_lr_ratio=config.min_learning_rate / max(1e-8, config.learning_rate),
        )

        # Dead feature tracking: counts tokens since each latent last fired
        self.tokens_since_activated = torch.zeros(
            dictionary_model.d_sae, dtype=torch.long, device=device
        )

        # WandB experiment tracking setup
        self._init_wandb()

    def _init_wandb(self):
        if HAS_WANDB and self.config.wandb_project:
            try:
                run_name = self.config.wandb_run_name or f"{self.model.__class__.__name__}_{int(time.time())}"
                wandb.init(
                    project=self.config.wandb_project,
                    name=run_name,
                    entity=self.config.wandb_entity,
                    config={
                        "architecture": self.model.__class__.__name__,
                        "d_in": self.model.d_in,
                        "d_sae": self.model.d_sae,
                        "d_out": self.model.d_out,
                        "config_kwargs": self.model.config_kwargs,
                        **self.config.model_dump(),
                    },
                    reinit=True,
                )
                logger.info(f"WandB initialized: project='{self.config.wandb_project}', run='{run_name}'")
            except Exception as e:
                logger.warning(f"WandB initialization failed/skipped: {e}")

    def train_step(self, batch: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        # Parse batch inputs
        if isinstance(batch, (tuple, list)):
            x_in = batch[0].to(self.device, non_blocking=True).contiguous()
            target = batch[1].to(self.device, non_blocking=True).contiguous()
        else:
            x_in = batch.to(self.device, non_blocking=True).contiguous()
            target = x_in

        tokens_in_batch = x_in.shape[0]
        self.total_tokens_trained += tokens_in_batch

        # Identify dead features (tokens without activation > threshold)
        dead_mask = self.tokens_since_activated > self.config.dead_feature_threshold

        # Forward pass
        out = self.model(x_in, target=target, dead_mask=dead_mask)
        total_loss = out.loss

        # Optional Ghost Gradients
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

        # Backward and step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update dead feature tracker
        with torch.no_grad():
            fired = (out.feature_acts > 0).any(dim=list(range(out.feature_acts.ndim - 1)))
            self.tokens_since_activated[fired] = 0
            self.tokens_since_activated[~fired] += tokens_in_batch

        # Compute metrics
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

        # Add architecture specific loss components
        for k, v in out.loss_dict.items():
            if isinstance(v, torch.Tensor):
                metrics[f"loss/{k}"] = v.item()

        return metrics

    def train(self) -> None:
        logger.info(f"Starting Training: {self.model.__class__.__name__} (d_in={self.model.d_in}, d_sae={self.model.d_sae})")
        logger.info(f"Target steps: {self.config.total_steps} | Batch size: {self.config.batch_size}")

        step = 0
        running_metrics: Dict[str, float] = {}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("Loss: {task.fields[loss]:.4f} | L0: {task.fields[l0]:.1f} | Dead: {task.fields[dead]:.1f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Training...",
                total=self.config.total_steps,
                loss=0.0,
                l0=0.0,
                dead=0.0,
            )

            for batch in self.buffer:
                step += 1
                metrics = self.train_step(batch)

                for k, v in metrics.items():
                    running_metrics[k] = 0.9 * running_metrics.get(k, v) + 0.1 * v

                # WandB live logging
                if HAS_WANDB and wandb.run is not None:
                    wandb.log(metrics, step=step)

                # Progress bar update
                progress.update(
                    task,
                    advance=1,
                    loss=running_metrics.get("loss/total", 0.0),
                    l0=running_metrics.get("metrics/l0", 0.0),
                    dead=running_metrics.get("metrics/dead_pct", 0.0),
                )

                # Checkpoint saving
                if step % self.config.checkpoint_steps == 0 or step == self.config.total_steps:
                    save_path = os.path.join(self.config.output_dir, f"step_{step}")
                    self.model.save_pretrained(
                        save_path,
                        metadata={
                            "step": step,
                            "tokens_processed": self.total_tokens_trained,
                            "metrics": running_metrics,
                        }
                    )
                    logger.info(f"Saved checkpoint to: {save_path} (step={step})")

                if step >= self.config.total_steps:
                    break

        logger.info("Training completed successfully!")
        if HAS_WANDB and wandb.run is not None:
            wandb.finish()
