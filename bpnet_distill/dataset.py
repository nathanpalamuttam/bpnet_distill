"""Dataset adapter that applies augmentations to each sample.

This is a thin wrapper around a base ``torch.utils.data.Dataset`` that yields
``(X,)`` or ``(X, label)`` or ``(X, X_ctl, label)`` batches. Augmentations are
applied to ``X`` only (see :mod:`bpnet_distill.augmentations` for the paired-
augmentation TODO).

Usage
-----
>>> base = SomeDataset(...)  # yields (X, label) or (X, X_ctl, label)
>>> ds = DistillationDataset(base, augmentations=[PointMutation(0.04)])
>>> loader = torch.utils.data.DataLoader(ds, batch_size=64, ...)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from .augmentations import Augmentation, Compose


BatchItem = Union[torch.Tensor, tuple]


class DistillationDataset(Dataset):
    """Wrap a base dataset and apply a list of augmentations to each ``X``.

    Parameters
    ----------
    base_dataset : torch.utils.data.Dataset
        Underlying dataset. Each item must be either ``X`` or a tuple whose
        first element is ``X`` (e.g., ``(X, label)`` or ``(X, X_ctl, label)``).
        The augmentation is applied to ``X`` only; other elements pass through
        unchanged.
    augmentations : sequence of callables or None
        Augmentations to apply in order. Each must be callable with signature
        ``(X) -> X``. If ``None`` or empty, the dataset is a pass-through.

    Notes
    -----
    Augmentations are applied inside ``__getitem__``, so with ``num_workers>0``
    the augmentation work happens in worker processes. If an augmentation
    object holds internal RNG state (e.g., :class:`PointMutation`), that state
    is *per-worker* — reseeding across workers is the user's responsibility.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        augmentations: Optional[Sequence[Augmentation]] = None,
    ) -> None:
        self.base_dataset = base_dataset
        if augmentations is None or len(augmentations) == 0:
            self._transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        else:
            self._transform = Compose(list(augmentations))

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> BatchItem:
        item = self.base_dataset[idx]

        if self._transform is None:
            return item

        # Single-tensor return
        if isinstance(item, torch.Tensor):
            return self._transform(item)

        # Tuple/list return: augment first element only.
        X = item[0]
        X_aug = self._transform(X)
        return (X_aug, *item[1:])