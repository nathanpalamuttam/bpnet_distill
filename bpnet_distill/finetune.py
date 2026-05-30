"""Supervised finetuning helpers for ChromBPNet/BPNet-style models.

The utilities in this module assume the common two-head forward contract used
by ChromBPNet and BPNet profile models:

``model(X) -> (profile_logits, log_counts)``.

They intentionally stay framework-light so users can finetune either full
models or distilled students without depending on a specific ChromBPNet
training repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .batch_utils import unpack_supervised_batch
from .losses import mnll_loss


@dataclass
class FineTuneConfig:
    """Configuration for :func:`finetune_chrombpnet`."""

    max_epochs: int = 10
    alpha: float = 1.0
    validation_iter: int = 100
    early_stop_epochs: int = 5
    grad_clip_norm: Optional[float] = None
    checkpoint_path: Optional[Union[str, Path]] = None
    verbose: bool = True


def set_trainable(
    model: torch.nn.Module,
    *,
    freeze: Optional[Sequence[str]] = None,
    unfreeze: Optional[Sequence[str]] = None,
) -> None:
    """Freeze or unfreeze parameters by name prefix.

    ``freeze=None`` leaves existing flags unchanged. ``freeze=[""]`` freezes
    the whole model. ``unfreeze`` is applied after ``freeze`` so callers can
    freeze a backbone and then unfreeze selected heads.
    """
    if freeze is not None:
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in freeze):
                param.requires_grad = False

    if unfreeze is not None:
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in unfreeze):
                param.requires_grad = True


def chrombpnet_supervised_loss(
    outputs,
    profile_counts: torch.Tensor,
    log_count_targets: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    *,
    alpha: float = 1.0,
    profile_loss_fn: Callable = mnll_loss,
    count_loss_fn: Callable = F.mse_loss,
) -> torch.Tensor:
    """Compute profile MNLL plus weighted log-count MSE.

    If explicit ``log_count_targets`` are not supplied, they are derived from
    the total observed profile counts as ``log(total + 1)``.
    """
    profile_logits, pred_log_counts = outputs
    profile_counts = profile_counts.to(
        device=profile_logits.device, dtype=profile_logits.dtype
    )

    if mask is not None:
        mask = mask.to(device=profile_logits.device, dtype=torch.bool)
        profile_counts = torch.where(
            mask,
            profile_counts,
            torch.zeros_like(profile_counts),
        )

    flat_logits = profile_logits.reshape(profile_logits.shape[0], -1)
    log_probs = torch.nn.functional.log_softmax(flat_logits, dim=-1)
    flat_counts = profile_counts.reshape(profile_counts.shape[0], -1)
    profile_loss = profile_loss_fn(log_probs, flat_counts).mean()

    if log_count_targets is None:
        totals = flat_counts.sum(dim=1, keepdim=True)
        log_count_targets = torch.log1p(totals)
    else:
        log_count_targets = log_count_targets.to(
            device=pred_log_counts.device, dtype=pred_log_counts.dtype
        )

    count_loss = count_loss_fn(pred_log_counts.squeeze(-1), log_count_targets.squeeze(-1))
    return profile_loss + float(alpha) * count_loss


def finetune_chrombpnet(
    model: torch.nn.Module,
    train_loader: Iterable,
    val_loader: Optional[Iterable],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    config: Optional[FineTuneConfig] = None,
    loss_fn: Optional[Callable] = None,
) -> List[Dict[str, float]]:
    """Finetune a ChromBPNet/BPNet-style model on observed profiles.

    Batches may be tuples or dictionaries. The required targets are observed
    profile counts; log-count targets are optional and otherwise derived from
    profile totals.
    """
    cfg = config or FineTuneConfig()
    loss_fn = loss_fn or chrombpnet_supervised_loss
    model.to(device)

    metrics: List[Dict[str, float]] = []
    best_val = float("inf")
    epochs_without_improvement = 0
    global_iter = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        running_loss = 0.0
        n_train = 0
        improved_this_epoch = False

        for batch in train_loader:
            X, y, log_counts, mask = unpack_supervised_batch(batch)
            X = X.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(
                model(X),
                y,
                log_counts,
                mask,
                alpha=cfg.alpha,
            )
            loss.backward()
            if cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            running_loss += float(loss.detach().cpu())
            n_train += 1

            if val_loader is not None and global_iter % cfg.validation_iter == 0:
                val_loss = evaluate_chrombpnet(
                    model, val_loader, device, loss_fn=loss_fn, alpha=cfg.alpha
                )
                if val_loss < best_val:
                    best_val = val_loss
                    epochs_without_improvement = 0
                    improved_this_epoch = True
                    if cfg.checkpoint_path is not None:
                        checkpoint_path = Path(cfg.checkpoint_path)
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(model.state_dict(), checkpoint_path)
                metric = {
                    "epoch": float(epoch),
                    "iter": float(global_iter),
                    "train_loss": float(loss.detach().cpu()),
                    "val_loss": float(val_loss),
                }
                metrics.append(metric)
                if cfg.verbose:
                    print(
                        "epoch={epoch} iter={iter} train_loss={train_loss:.6f} "
                        "val_loss={val_loss:.6f}".format(**metric)
                    )

            global_iter += 1

        if val_loader is None:
            metrics.append(
                {
                    "epoch": float(epoch),
                    "iter": float(global_iter),
                    "train_loss": running_loss / max(n_train, 1),
                    "val_loss": float("nan"),
                }
            )
        else:
            if not improved_this_epoch:
                epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stop_epochs:
                if cfg.verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

    return metrics


@torch.no_grad()
def evaluate_chrombpnet(
    model: torch.nn.Module,
    loader: Iterable,
    device: torch.device,
    *,
    loss_fn: Callable = chrombpnet_supervised_loss,
    alpha: float = 1.0,
) -> float:
    """Return mean supervised loss over a validation loader."""
    model.eval()
    total = 0.0
    n_batches = 0
    for batch in loader:
        X, y, log_counts, mask = unpack_supervised_batch(batch)
        X = X.to(device=device, dtype=torch.float32)
        loss = loss_fn(model(X), y, log_counts, mask, alpha=alpha)
        total += float(loss.detach().cpu())
        n_batches += 1
    if n_batches == 0:
        raise RuntimeError("Validation loader yielded zero batches.")
    return total / n_batches
