"""Main user-facing trainer class for profile+count distillation.

:class:`DistillationTrainer` orchestrates:
  * a student model with two heads (profile logits, log counts),
  * a :class:`TeacherEnsemble` that produces distillation targets,
  * training/validation data loaders,
  * default or user-supplied train-step and eval-step callables,
  * default loss callables (MNLL + alpha * MSE) with TODO for unified override.

The trainer delegates the actual optimization loop to
``Model.fit_generator`` from ``BPNet_strand_merged_umap`` (per the design
decision to keep that loop). Custom ``train_step_fn`` / ``eval_step_fn``
callables are accepted and used **instead of** ``fit_generator`` when both
are provided, giving users an escape hatch.

Public API
----------
>>> trainer = DistillationTrainer(
...     student=student_model,
...     teachers=[t1, t2, t3],
...     device=torch.device("cuda"),
...     alpha=1.0,
...     teacher_device_policy="swap_per_batch",
... )
>>> trainer.fit(
...     train_loader=train_loader,
...     val_loader=val_loader,
...     optimizer=optimizer,
...     max_epochs=100,
...     n_val_batches=10,
...     validation_iter=100,
...     early_stop_epochs=10,
... )

TODO
----
- Expose hooks for LR scheduling, gradient clipping, mixed precision, and
  checkpointing beyond what ``fit_generator`` provides.
- Accept a unified loss callable (see :mod:`bpnet_distill.losses`).
- Accept a ``Dataset`` and build the loader internally (see
  :mod:`bpnet_distill.dataset`).
- Richer batch protocols beyond ``X``-only (see
  :mod:`bpnet_distill.generators`).
- Pluggable teacher aggregation (see :mod:`bpnet_distill.teacher`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from .generators import TeacherDistillationGenerator, build_validation_arrays
from .losses import log_count_mse_loss, mnll_loss
from .teacher import DevicePolicy, TeacherEnsemble


TrainStepFn = Callable[..., float]
EvalStepFn = Callable[..., float]


class DistillationTrainer:
    """Orchestrate distillation training of a profile+count student model.

    Parameters
    ----------
    student : torch.nn.Module
        Student model. Must implement ``student(X) -> (profile_logits,
        log_counts)``. To use the default training loop the student must also
        be an instance of ``BPNet_strand_merged_umap.Model`` (exposes
        ``fit_generator``); otherwise supply ``train_step_fn`` and
        ``eval_step_fn``.
    teachers : sequence of torch.nn.Module
        Teacher models. Assumed to share the student's architecture (same
        input and output shapes). A TODO in :mod:`bpnet_distill.teacher`
        covers differing-architecture support.
    device : torch.device
        Device on which student training runs; also the inference device for
        teachers when ``teacher_device_policy`` is not ``"cpu_only"``.
    alpha : float, optional
        Weight on the count loss when using the default loss formulation
        (``mnll + alpha * mse``). Default ``1.0``.
    profile_loss_fn : callable, optional
        ``(log_probs, true_counts) -> tensor`` per-example profile loss.
        Default :func:`bpnet_distill.losses.mnll_loss`.
    count_loss_fn : callable, optional
        ``(pred_log_counts, true_counts) -> scalar`` count loss. Default
        :func:`bpnet_distill.losses.log_count_mse_loss`.
    teacher_device_policy : {"all_on_device", "swap_per_batch", "cpu_only"}
        See :class:`TeacherEnsemble`. Default ``"swap_per_batch"``.
    train_step_fn, eval_step_fn : callable, optional
        Advanced escape hatch. If both are provided, :meth:`fit` bypasses
        ``fit_generator`` and runs a custom loop that calls these per batch.
        Signatures::

            train_step_fn(student, batch, optimizer, device) -> float
            eval_step_fn(student, batch, device) -> float

        where ``batch`` is ``(X, y, mask)`` as produced by
        :class:`TeacherDistillationGenerator`.
    """

    def __init__(
        self,
        student: torch.nn.Module,
        teachers: Sequence[torch.nn.Module],
        device: torch.device,
        *,
        alpha: float = 1.0,
        profile_loss_fn: Callable = mnll_loss,
        count_loss_fn: Callable = log_count_mse_loss,
        teacher_device_policy: DevicePolicy = "swap_per_batch",
        train_step_fn: Optional[TrainStepFn] = None,
        eval_step_fn: Optional[EvalStepFn] = None,
    ) -> None:
        self.student = student.to(device)
        self.device = device
        self.alpha = float(alpha)
        self.profile_loss_fn = profile_loss_fn
        self.count_loss_fn = count_loss_fn

        self.teachers = TeacherEnsemble(
            models=teachers,
            device=device,
            device_policy=teacher_device_policy,
        )

        self.train_step_fn = train_step_fn
        self.eval_step_fn = eval_step_fn

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        *,
        max_epochs: int = 100,
        batch_size: int = 64,
        n_val_batches: int = 10,
        validation_iter: int = 100,
        early_stop_epochs: int = 10,
        verbose: bool = True,
        save: bool = True,
        metrics_path: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """Train the student.

        Parameters
        ----------
        train_loader : Iterable
            Training loader. Each batch must be ``X`` or a tuple whose first
            element is ``X``. Augmentations should be baked into the loader's
            dataset via :class:`DistillationDataset`.
        val_loader : Iterable
            Validation loader. Teachers run over ``n_val_batches`` of this
            loader once to build a fixed validation set.
        optimizer : torch.optim.Optimizer
            Optimizer for the student's parameters.
        max_epochs : int, optional
            Maximum number of epochs. Default 100.
        batch_size : int, optional
            Batch size passed through to ``fit_generator``. Default 64.
        n_val_batches : int, optional
            Number of batches drawn from ``val_loader`` to build the fixed
            validation arrays. Default 10.
        validation_iter : int, optional
            Validate every N iterations (see ``fit_generator``). Default 100.
        early_stop_epochs : int, optional
            Stop if no validation improvement for this many epochs.
        verbose : bool, optional
            Per-epoch logging.
        save : bool, optional
            Passed through to ``fit_generator``; writes best checkpoint to the
            student's ``model_save_path``.
        metrics_path : str or Path, optional
            If given, training metrics are also written to this path as TSV
            after training completes.

        Returns
        -------
        list of str
            The ``train_metrics`` list from the student (TSV-formatted lines).
        """
        # Build fixed validation set
        if verbose:
            print(f"Building validation set ({n_val_batches} batches)...")
        X_valid, y_valid = build_validation_arrays(
            val_loader, self.teachers, self.device, n_val_batches
        )
        if verbose:
            print(f"Validation set: {X_valid.shape[0]} examples")

        training_data = TeacherDistillationGenerator(
            loader=train_loader,
            teachers=self.teachers,
            device=self.device,
        )

        # Escape hatch: custom loop
        if self.train_step_fn is not None and self.eval_step_fn is not None:
            metrics = self._custom_fit(
                training_data=training_data,
                X_valid=X_valid,
                y_valid=y_valid,
                optimizer=optimizer,
                max_epochs=max_epochs,
                validation_iter=validation_iter,
                early_stop_epochs=early_stop_epochs,
                verbose=verbose,
            )
        else:
            metrics = self._fit_generator(
                training_data=training_data,
                X_valid=X_valid,
                y_valid=y_valid,
                optimizer=optimizer,
                max_epochs=max_epochs,
                batch_size=batch_size,
                validation_iter=validation_iter,
                early_stop_epochs=early_stop_epochs,
                verbose=verbose,
                save=save,
            )

        if metrics_path is not None and metrics:
            metrics_path = Path(metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("w") as handle:
                for line in metrics:
                    handle.write(line + "\n")
            if verbose:
                print(f"Saved metrics to {metrics_path}")

        return metrics

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #

    def _fit_generator(
        self,
        *,
        training_data: TeacherDistillationGenerator,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        batch_size: int,
        validation_iter: int,
        early_stop_epochs: int,
        verbose: bool,
        save: bool,
    ) -> List[str]:
        """Delegate to ``Model.fit_generator`` from BPNet_strand_merged_umap."""
        if not hasattr(self.student, "fit_generator"):
            raise AttributeError(
                "Default training loop requires the student to expose "
                "``fit_generator`` (typically a BPNet_strand_merged_umap.Model "
                "instance). Provide ``train_step_fn`` and ``eval_step_fn`` to "
                "use a custom loop instead."
            )

        self.student.fit_generator(
            training_data=training_data,
            optimizer=optimizer,
            X_valid=X_valid,
            y_valid=y_valid,
            max_epochs=max_epochs,
            batch_size=batch_size,
            validation_iter=validation_iter,
            early_stop_epochs=early_stop_epochs,
            verbose=verbose,
            save=save,
        )
        return list(getattr(self.student, "train_metrics", []) or [])

    def _custom_fit(
        self,
        *,
        training_data: TeacherDistillationGenerator,
        X_valid: np.ndarray,
        y_valid: np.ndarray,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        validation_iter: int,
        early_stop_epochs: int,
        verbose: bool,
    ) -> List[str]:
        """Run user-supplied train/eval step callables.

        This is a minimal loop intended for users who want full control. It
        does not implement the full bpnet-lite feature set (mixed precision,
        LR scheduling, complex checkpointing) — that is the TODO noted above.
        """
        assert self.train_step_fn is not None
        assert self.eval_step_fn is not None

        metrics: List[str] = ["epoch\titer\ttrain_loss\tval_loss"]
        best_val = float("inf")
        epochs_since_improvement = 0

        X_valid_t = torch.from_numpy(X_valid)
        y_valid_t = torch.from_numpy(y_valid)
        mask_valid_t = torch.ones_like(y_valid_t, dtype=torch.bool)

        global_iter = 0
        for epoch in range(max_epochs):
            self.student.train()
            for batch in training_data:
                train_loss = self.train_step_fn(
                    self.student, batch, optimizer, self.device
                )

                if global_iter % validation_iter == 0:
                    self.student.eval()
                    val_batch = (X_valid_t, y_valid_t, mask_valid_t)
                    with torch.no_grad():
                        val_loss = self.eval_step_fn(
                            self.student, val_batch, self.device
                        )
                    metrics.append(
                        f"{epoch}\t{global_iter}\t{train_loss:.6f}\t{val_loss:.6f}"
                    )
                    if verbose:
                        print(metrics[-1])

                    if val_loss < best_val:
                        best_val = val_loss
                        epochs_since_improvement = 0
                    self.student.train()

                global_iter += 1

            epochs_since_improvement += 1
            if epochs_since_improvement >= early_stop_epochs:
                if verbose:
                    print(f"Early stopping at epoch {epoch}.")
                break

        return metrics