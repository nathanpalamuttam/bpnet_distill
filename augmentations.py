"""Sequence augmentations for distillation training.

Each augmentation is a callable that takes a one-hot encoded tensor of shape
(4, length) and returns a tensor of the same shape (or a padded/cropped tensor
of length `in_window` for length-changing augmentations like structural
variations).

Augmentations are composed via :class:`Compose` and applied inside the
``Dataset`` (see :mod:`bpnet_distill.dataset`). This keeps augmentation work in
worker processes when ``num_workers > 0``.

Usage
-----
>>> from bpnet_distill.augmentations import Compose, PointMutation, ReverseComplement
>>> aug = Compose([PointMutation(rate=0.04), ReverseComplement(p=0.5)])
>>> X_aug = aug(X)  # X: torch.Tensor, shape (4, L)

Notes
-----
Currently augmentations only transform the input sequence ``X``. Teacher
targets are generated from the augmented ``X`` at train time, so teacher
outputs implicitly match the augmented input (consistent with the original
distillation loop this package was extracted from).

TODO
----
- Support paired augmentations that transform a target signal alongside the
  input (e.g., reverse-complement flipping both strands of a profile target).
  The current contract is single-argument ``__call__(X) -> X``; paired support
  would require a two-argument variant or a richer batch protocol.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch


class Augmentation:
    """Base class for sequence augmentations.

    Subclasses implement :meth:`__call__` taking a one-hot encoded tensor of
    shape ``(4, L)`` and returning a tensor of shape ``(4, L')``. Most
    augmentations preserve length; length-changing augmentations must restore
    ``L' == in_window`` (see :class:`StructuralVariation`).

    Subclasses accept an optional ``random_state`` (int or
    ``numpy.random.RandomState``) at construction to ensure deterministic
    behaviour. If ``None``, a fresh random state is used.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self.random_state = self._resolve_random_state(random_state)

    @staticmethod
    def _resolve_random_state(
        random_state: Optional[int],
    ) -> np.random.RandomState:
        if isinstance(random_state, np.random.RandomState):
            return random_state
        return np.random.RandomState(random_state)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class Compose(Augmentation):
    """Apply a sequence of augmentations in order.

    Parameters
    ----------
    augmentations : sequence of callables
        Each element must be callable with signature ``(X) -> X``.
    """

    def __init__(self, augmentations: Sequence[Augmentation]) -> None:
        # Compose itself has no randomness; children manage their own.
        self.augmentations: List[Augmentation] = list(augmentations)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            X = aug(X)
        return X

    def __repr__(self) -> str:
        inner = ", ".join(repr(a) for a in self.augmentations)
        return f"Compose([{inner}])"


class PointMutation(Augmentation):
    """Randomly mutate individual nucleotides.

    Each position is mutated independently with probability ``rate``. A
    mutated position is resampled uniformly from the four nucleotides (so a
    mutation may re-select the original base; this matches the behaviour of
    the reference dataloader).

    Parameters
    ----------
    rate : float
        Per-position mutation probability. ``0.0`` disables the augmentation.
    random_state : int or None, optional
        Seed for deterministic mutations.
    """

    def __init__(self, rate: float = 0.04, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)
        self.rate = float(rate)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self.rate <= 0.0:
            return X

        length = X.shape[1]
        mutation_mask = self.random_state.uniform(size=length) < self.rate
        if not mutation_mask.any():
            return X

        X_np = X.numpy().copy()
        positions = np.where(mutation_mask)[0]
        for pos in positions:
            X_np[:, pos] = 0
            new_nuc = self.random_state.randint(0, 4)
            X_np[new_nuc, pos] = 1
        return torch.from_numpy(X_np)

    def __repr__(self) -> str:
        return f"PointMutation(rate={self.rate})"


class StructuralVariation(Augmentation):
    """Apply random structural variations (insertion, deletion, inversion).

    The number of SVs per call is sampled from ``Poisson(rate)``. Each SV has
    a length uniform in ``[min_length, max_length]`` and a type uniform over
    ``{insertion, deletion, inversion}``. Inversions are reverse-complemented
    (flip along positions and swap A<->T, C<->G assuming channel order
    ``[A, C, G, T]``).

    After all SVs are applied, the output is center-cropped or zero-padded to
    ``in_window``.

    Parameters
    ----------
    rate : float
        Poisson lambda for number of SVs per sequence. ``0.0`` disables.
    min_length, max_length : int
        Inclusive range for SV length in bp.
    in_window : int
        Target output length after length-changing SVs.
    random_state : int or None, optional
    """

    def __init__(
        self,
        rate: float = 1.0,
        min_length: int = 1,
        max_length: int = 20,
        in_window: int = 2114,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(random_state=random_state)
        self.rate = float(rate)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.in_window = int(in_window)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self.rate <= 0.0:
            return X

        n_svs = self.random_state.poisson(self.rate)
        if n_svs == 0:
            return X

        X_np = X.numpy().copy()
        seq_length = X_np.shape[1]

        for _ in range(n_svs):
            sv_type = self.random_state.choice(["insertion", "deletion", "inversion"])
            sv_length = self.random_state.randint(self.min_length, self.max_length + 1)

            if sv_type == "insertion":
                pos = self.random_state.randint(0, seq_length)
                random_seq = np.zeros((4, sv_length), dtype=X_np.dtype)
                random_nucs = self.random_state.randint(0, 4, size=sv_length)
                random_seq[random_nucs, np.arange(sv_length)] = 1
                X_np = np.concatenate([X_np[:, :pos], random_seq, X_np[:, pos:]], axis=1)
                seq_length += sv_length

            elif sv_type == "deletion":
                if seq_length <= sv_length:
                    continue
                pos = self.random_state.randint(0, seq_length - sv_length + 1)
                X_np = np.concatenate(
                    [X_np[:, :pos], X_np[:, pos + sv_length:]], axis=1
                )
                seq_length -= sv_length

            elif sv_type == "inversion":
                if seq_length <= sv_length:
                    continue
                pos = self.random_state.randint(0, seq_length - sv_length + 1)
                region = X_np[:, pos:pos + sv_length]
                region = np.flip(region, axis=1).copy()
                # Complement assumes channel order [A, C, G, T]: A<->T, C<->G.
                region = region[[3, 2, 1, 0], :]
                X_np[:, pos:pos + sv_length] = region

        # Restore target length by center crop / zero pad.
        if X_np.shape[1] > self.in_window:
            excess = X_np.shape[1] - self.in_window
            start = excess // 2
            X_np = X_np[:, start:start + self.in_window]
        elif X_np.shape[1] < self.in_window:
            deficit = self.in_window - X_np.shape[1]
            pad_left = deficit // 2
            pad_right = deficit - pad_left
            X_np = np.pad(
                X_np, ((0, 0), (pad_left, pad_right)),
                mode="constant", constant_values=0,
            )

        return torch.from_numpy(X_np)

    def __repr__(self) -> str:
        return (
            f"StructuralVariation(rate={self.rate}, "
            f"min_length={self.min_length}, max_length={self.max_length}, "
            f"in_window={self.in_window})"
        )


class ReverseComplement(Augmentation):
    """Reverse-complement the sequence with probability ``p``.

    Assumes channel order ``[A, C, G, T]``. Reverse-complement is implemented
    as ``torch.flip(X, dims=[0, 1])`` which flips both the channel axis (A<->T,
    C<->G) and the position axis.

    Parameters
    ----------
    p : float
        Probability of applying the flip. ``0.5`` matches the original loop.
    random_state : int or None, optional
    """

    def __init__(self, p: float = 0.5, random_state: Optional[int] = None) -> None:
        super().__init__(random_state=random_state)
        self.p = float(p)

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return X
        if self.random_state.uniform() < self.p:
            return torch.flip(X, dims=[0, 1])
        return X

    def __repr__(self) -> str:
        return f"ReverseComplement(p={self.p})"