"""Batch parsing helpers shared by training and benchmark utilities."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import torch


def unpack_x(batch: Any) -> torch.Tensor:
    """Extract the sequence tensor from common dataloader batch formats."""
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, Mapping):
        for key in ("X", "x", "sequence", "sequences", "inputs"):
            if key in batch:
                return batch[key]
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        return batch[0]
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def unpack_supervised_batch(
    batch: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract ``X``, profile counts, optional log-counts, and optional mask.

    Supported forms are:

    - ``(X, profile_counts)``
    - ``(X, profile_counts, log_counts)``
    - ``(X, profile_counts, log_counts, mask)``
    - dictionaries with ``X`` plus profile/count/mask keys
    """
    if isinstance(batch, Mapping):
        X = unpack_x(batch)
        profile = None
        for key in ("profile_counts", "profile", "y", "Y", "counts"):
            if key in batch:
                profile = batch[key]
                break
        if profile is None:
            raise KeyError(
                "Batch dictionary must include one of: profile_counts, "
                "profile, y, Y, counts."
            )

        log_counts = None
        for key in ("log_counts", "logcounts", "log_count_targets"):
            if key in batch:
                log_counts = batch[key]
                break

        mask = batch.get("mask")
        return X, profile, log_counts, mask

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Supervised batches must include at least X and y.")
        X = batch[0]
        profile = batch[1]
        log_counts = batch[2] if len(batch) >= 3 else None
        mask = batch[3] if len(batch) >= 4 else None
        return X, profile, log_counts, mask

    raise TypeError(f"Unsupported supervised batch type: {type(batch)!r}")
