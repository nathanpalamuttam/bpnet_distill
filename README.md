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

## ChromBPNet finetuning

ChromBPNet models use the same two-head profile/count contract as BPNet-style
models:

```python
profile_logits, log_counts = model(X)
```

Use `finetune_chrombpnet` when you want to continue training a full
ChromBPNet model, a ChromBPNet-initialized student, or a distilled student on
observed profile counts:

```python
import torch
from bpnet_distill import FineTuneConfig, finetune_chrombpnet, set_trainable

# Optional: freeze a backbone and finetune only profile/count heads. Prefixes
# are matched against `model.named_parameters()`.
set_trainable(
    model,
    freeze=[""],
    unfreeze=["profile_head", "count_head"],
)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
    weight_decay=1e-4,
)

metrics = finetune_chrombpnet(
    model=model,
    train_loader=train_loader,  # yields (X, profile_counts[, log_counts, mask])
    val_loader=val_loader,
    optimizer=optimizer,
    device=torch.device("cuda"),
    config=FineTuneConfig(
        max_epochs=20,
        alpha=0.5,
        validation_iter=200,
        early_stop_epochs=5,
        checkpoint_path="checkpoints/chrombpnet_finetuned.pt",
    ),
)
```

If a batch does not include explicit log-count targets, the finetuning loss
derives them as `log(total_profile_counts + 1)`. Dictionary batches are also
supported with sequence keys like `X`/`sequence` and profile keys like
`profile_counts`/`counts`.

## QTL and GWAS benchmarks

The ChromBPNet preprint benchmarks variant predictions with allelic log-count
effects, profile divergence, QTL classification/correlation, and enrichment
among fine-mapped GWAS variants. This package exposes those pieces as small
utilities so the same workflow can be applied to ChromBPNet or distilled
BPNet-style models.

```python
from bpnet_distill import (
    VariantExample,
    effect_correlation,
    gwas_enrichment_by_threshold,
    qtl_classification_metrics,
    score_variant_effects,
)

variants = [
    VariantExample(
        variant_id="chr1:1000:A:G",
        ref_sequence=ref_one_hot,  # shape (4, length)
        alt_sequence=alt_one_hot,  # shape (4, length)
        label=1,                   # optional QTL/non-QTL label
        observed_effect=0.42,       # optional QTL effect size
        pip=0.76,                  # optional fine-mapping posterior
        locus_id="trait_locus_1",
    ),
]

rows = score_variant_effects(model, variants, torch.device("cuda"))
qtl_metrics = qtl_classification_metrics(rows, score_key="abs_log_count_delta")
effect_metrics = effect_correlation(rows, predicted_key="log_count_delta")
gwas_enrichment = gwas_enrichment_by_threshold(
    rows,
    score_thresholds=(0.25, 0.5, 1.0),
    pip_thresholds=(0.1, 0.5, 0.9),
)
```

`score_variant_effects` returns `log_count_delta` (`alt - ref`),
`abs_log_count_delta`, and `profile_jsd` for each variant while carrying
through labels, observed effects, PIPs, and locus IDs when provided.

## Project layout

```
bpnet-distill/
├── pyproject.toml
├── README.md
└── bpnet_distill/
    ├── __init__.py
    ├── augmentations.py
    ├── batch_utils.py
    ├── benchmarks.py
    ├── dataset.py
    ├── finetune.py
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
