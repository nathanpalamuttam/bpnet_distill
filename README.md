# bpnet-distill

Distillation trainer for profile+count models (BPNet-style).

## Install

From the project root (where `pyproject.toml` lives):

```bash
pip install -e .
```

`-e` means "editable" — changes you make to the source are picked up
immediately without reinstalling.

Then from anywhere in any script:

```python
from bpnet_distill import DistillationTrainer, DistillationDataset, PointMutation
```

## Quick start

```python
import torch
from bpnet_distill import (
    DistillationTrainer,
    DistillationDataset,
    PointMutation,
    StructuralVariation,
    ReverseComplement,
)

# 1. Wrap your existing dataset with augmentations
train_ds = DistillationDataset(
    base_train_ds,
    augmentations=[
        PointMutation(rate=0.04),
        StructuralVariation(rate=1.0, in_window=2114),
        ReverseComplement(p=0.5),
    ],
)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=64)

# 2. Build the trainer
trainer = DistillationTrainer(
    student=student_model,
    teachers=[t1, t2, t3],
    device=torch.device("cuda"),
    alpha=1.0,
    teacher_device_policy="swap_per_batch",
)

# 3. Fit
trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=torch.optim.Adam(student_model.parameters(), lr=1e-3),
    max_epochs=100,
    n_val_batches=10,
    validation_iter=100,
    early_stop_epochs=10,
)
```

## Project layout

```
bpnet-distill/
├── pyproject.toml
├── README.md
└── bpnet_distill/
    ├── __init__.py
    ├── augmentations.py
    ├── dataset.py
    ├── generators.py
    ├── losses.py
    ├── teacher.py
    └── trainer.py
```

## Notes

The default training loop delegates to `Model.fit_generator` from
`BPNet_strand_merged_umap`, which is part of your existing project and is
**not** declared as a dependency here. Make sure it is importable in the
environment where you run training. To use a different student model, pass
`train_step_fn` and `eval_step_fn` to `DistillationTrainer`.
