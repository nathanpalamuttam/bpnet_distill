"""bpnet_distill: a distillation trainer for profile+count models.

Public API
----------
Core trainer:
    :class:`DistillationTrainer` — main orchestration class.

Teacher management:
    :class:`TeacherEnsemble` — wraps a list of teacher models with a device
    placement policy and fixed aggregation.

Data:
    :class:`DistillationDataset` — wraps a base dataset, applies augmentations.
    :class:`TeacherDistillationGenerator` — yields ``(X, y, mask)`` for
    ``fit_generator``.
    :func:`build_validation_arrays` — materialize a fixed validation set.

ChromBPNet finetuning:
    :func:`finetune_chrombpnet` — supervised finetuning loop for two-head
    profile models.
    :class:`FineTuneConfig` — finetuning configuration.

Variant benchmarks:
    :class:`VariantExample` — ref/alt sequence pair plus benchmark metadata.
    :func:`score_variant_effects` — count/profile variant-effect scoring.
    :func:`qtl_classification_metrics` — QTL AP/AUC benchmark.
    :func:`effect_correlation` — observed vs predicted effect correlation.
    :func:`gwas_enrichment_by_threshold` — fine-mapped GWAS enrichment.

Augmentations (composable callables):
    :class:`Compose`, :class:`PointMutation`, :class:`StructuralVariation`,
    :class:`ReverseComplement`.

Default losses:
    :func:`mnll_loss`, :func:`log_count_mse_loss`.

Usage
-----
>>> import torch
>>> from bpnet_distill import (
...     DistillationTrainer, DistillationDataset,
...     PointMutation, StructuralVariation, ReverseComplement,
... )
>>>
>>> train_ds = DistillationDataset(
...     base_train_ds,
...     augmentations=[
...         PointMutation(rate=0.04),
...         StructuralVariation(rate=1.0, in_window=2114),
...         ReverseComplement(p=0.5),
...     ],
... )
>>> train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64)
>>> val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)
>>>
>>> trainer = DistillationTrainer(
...     student=student_model,
...     teachers=[teacher_1, teacher_2, teacher_3],
...     device=torch.device("cuda"),
...     alpha=1.0,
... )
>>> trainer.fit(train_loader, val_loader, optimizer, max_epochs=100)
"""

from .augmentations import (
    Augmentation,
    Compose,
    PointMutation,
    ReverseComplement,
    StructuralVariation,
)
from .dataset import DistillationDataset
from .benchmarks import (
    VariantExample,
    effect_correlation,
    gwas_enrichment_by_threshold,
    qtl_classification_metrics,
    score_variant_effects,
)
from .finetune import (
    FineTuneConfig,
    chrombpnet_supervised_loss,
    evaluate_chrombpnet,
    finetune_chrombpnet,
    set_trainable,
)
from .generators import (
    TeacherDistillationGenerator,
    build_validation_arrays,
)
from .losses import log_count_mse_loss, mnll_loss
from .teacher import DevicePolicy, TeacherEnsemble
from .trainer import DistillationTrainer

__all__ = [
    # Trainer
    "DistillationTrainer",
    # Teacher
    "TeacherEnsemble",
    "DevicePolicy",
    # Data
    "DistillationDataset",
    "TeacherDistillationGenerator",
    "build_validation_arrays",
    # ChromBPNet finetuning
    "FineTuneConfig",
    "chrombpnet_supervised_loss",
    "evaluate_chrombpnet",
    "finetune_chrombpnet",
    "set_trainable",
    # Variant benchmarks
    "VariantExample",
    "score_variant_effects",
    "qtl_classification_metrics",
    "effect_correlation",
    "gwas_enrichment_by_threshold",
    # Augmentations
    "Augmentation",
    "Compose",
    "PointMutation",
    "StructuralVariation",
    "ReverseComplement",
    # Losses
    "mnll_loss",
    "log_count_mse_loss",
]
