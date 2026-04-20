"""Generators and helpers for producing teacher-distilled targets on the fly.

:class:`TeacherDistillationGenerator` wraps a base data loader and produces
``(X, y, mask)`` batches suitable for ``Model.fit_generator`` from
``BPNet_strand_merged_umap``. It runs the teacher ensemble on each batch and
post-processes the targets (integer rounding, zero-sum safeguard) to match the
behaviour of the original distillation loop.

:func:`build_validation_arrays` materializes a fixed validation set by running
the teachers over ``n_batches`` of a validation loader once and concatenating
results into numpy arrays, matching the validation contract of
``fit_generator``.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Tuple

import numpy as np
import torch

from .teacher import TeacherEnsemble


def _unpack_batch(batch) -> torch.Tensor:
    """Extract ``X`` from a batch tuple.

    Supports ``X``, ``(X,)``, ``(X, label)``, ``(X, X_ctl, label)``. For v1
    only ``X`` is used; additional elements are silently dropped.

    TODO: richer batch protocols (e.g., dicts, ``(X, y, mask)`` passthrough).
    """
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (list, tuple)):
        return batch[0]
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def _postprocess_teacher_targets(profile_counts: torch.Tensor) -> torch.Tensor:
    """Round to integers and guarantee non-zero totals for each example.

    ``Model.fit_generator`` computes MNLL which requires positive total counts
    per example. Examples with zero total after rounding get a single count
    placed at position ``(0, 0)``.
    """
    profile_counts = torch.round(profile_counts)
    totals = profile_counts.reshape(profile_counts.shape[0], -1).sum(dim=1)
    zero_mask = totals == 0
    if zero_mask.any():
        profile_counts[zero_mask, 0, 0] = 1.0
    return profile_counts


class TeacherDistillationGenerator:
    """Iterable wrapper yielding ``(X, y, mask)`` for ``Model.fit_generator``.

    Parameters
    ----------
    loader : Iterable
        Base data loader yielding ``X`` or tuples whose first element is ``X``.
    teachers : TeacherEnsemble
        Teacher ensemble that produces distillation targets.
    device : torch.device
        Device on which teacher inference runs.

    Yields
    ------
    X : torch.Tensor, shape (batch, 4, in_window)
        Input batch (on CPU, as ``fit_generator`` expects).
    y : torch.Tensor, shape (batch, n_outputs, out_window)
        Teacher-generated profile counts (rounded to integers).
    mask : torch.Tensor, same shape as y
        All-ones boolean mask. Present for API compatibility with
        ``fit_generator``.
    """

    def __init__(
        self,
        loader: Iterable,
        teachers: TeacherEnsemble,
        device: torch.device,
    ) -> None:
        self.loader = loader
        self.teachers = teachers
        self.device = device

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for batch in self.loader:
            X_cpu = _unpack_batch(batch).to(dtype=torch.float32)

            if self.teachers.device_policy == "cpu_only":
                X_teacher = X_cpu
            else:
                X_teacher = X_cpu.to(self.device, non_blocking=True)

            profile_counts, _ = self.teachers.predict(X_teacher)
            profile_counts = _postprocess_teacher_targets(profile_counts).cpu()
            mask = torch.ones_like(profile_counts, dtype=torch.bool)

            yield X_cpu, profile_counts, mask


def build_validation_arrays(
    loader: Iterable,
    teachers: TeacherEnsemble,
    device: torch.device,
    n_batches: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Materialize a fixed validation set by running teachers over ``n_batches``.

    Parameters
    ----------
    loader : Iterable
        Validation loader (typically configured with no augmentations so the
        validation target is stable).
    teachers : TeacherEnsemble
    device : torch.device
    n_batches : int
        Number of batches to draw from ``loader``.

    Returns
    -------
    X_valid : np.ndarray, shape (N, 4, in_window), dtype float32
    y_valid : np.ndarray, shape (N, n_outputs, out_window), dtype float32

    Notes
    -----
    The original loop materialises validation once at the start of training
    and reuses the same arrays across epochs. This matches that behaviour.
    """
    X_list, y_list = [], []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        X_cpu = _unpack_batch(batch).to(dtype=torch.float32)
        if teachers.device_policy == "cpu_only":
            X_teacher = X_cpu
        else:
            X_teacher = X_cpu.to(device, non_blocking=True)

        profile_counts, _ = teachers.predict(X_teacher)
        profile_counts = _postprocess_teacher_targets(profile_counts).cpu()

        X_list.append(X_cpu.numpy())
        y_list.append(profile_counts.numpy())

    if not X_list:
        raise RuntimeError(
            "Validation loader yielded zero batches; cannot build validation set."
        )

    X_valid = np.concatenate(X_list, axis=0).astype(np.float32)
    y_valid = np.concatenate(y_list, axis=0).astype(np.float32)
    return X_valid, y_valid