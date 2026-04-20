"""Teacher ensemble management: device placement and target generation.

This module defines :class:`TeacherEnsemble`, which wraps a list of teacher
models and produces aggregated ``(profile_counts, log_counts)`` targets for
each input batch.

Teachers are assumed to share the student's architecture (two-head:
``model(X) -> (profile_logits, log_counts)``). A TODO below notes the need
for architecture adapters in a future version.

Device policies
---------------
Teacher inference is the dominant cost in distillation. The policy controls
the tradeoff between GPU memory and per-batch overhead:

- ``"all_on_device"``: move every teacher to ``device`` once at construction
  and leave them there. Fastest per batch; highest memory.
- ``"swap_per_batch"``: teachers live on CPU; each batch moves one teacher
  at a time to GPU, runs inference, moves it back. Slowest per batch; lowest
  GPU memory. Matches the original loop.
- ``"cpu_only"``: teachers stay on CPU and inference happens on CPU regardless
  of ``device``. Useful for tiny GPUs or debugging.

TODO
----
- Pluggable teacher aggregation. Currently fixed: average ``probs * total``
  across teachers, derive totals by summing and adding 1.0, then take log.
  A future ``aggregator`` callable should accept a list of
  ``(profile_logits, log_counts)`` pairs and return aggregated
  ``(profile_counts, log_counts)``.
- Support teachers with differing architectures (e.g., different output
  window, different number of tracks) via an adapter callable.
"""

from __future__ import annotations

from typing import List, Literal, Sequence, Tuple

import torch


DevicePolicy = Literal["all_on_device", "swap_per_batch", "cpu_only"]


class TeacherEnsemble:
    """Ensemble of distillation teacher models.

    Parameters
    ----------
    models : sequence of torch.nn.Module
        Teacher models. Each must support the same forward contract as the
        student: ``model(X) -> (profile_logits, log_counts)`` where
        ``profile_logits`` has shape ``(batch, n_outputs, out_window)`` and
        ``log_counts`` has shape ``(batch, n_outputs)`` (or ``(batch, 1)``).
    device : torch.device
        Inference device.
    device_policy : {"all_on_device", "swap_per_batch", "cpu_only"}
        See module docstring. Default ``"swap_per_batch"`` matches the
        original distillation loop.
    """

    def __init__(
        self,
        models: Sequence[torch.nn.Module],
        device: torch.device,
        device_policy: DevicePolicy = "swap_per_batch",
    ) -> None:
        if len(models) == 0:
            raise ValueError("TeacherEnsemble requires at least one model.")

        self.models: List[torch.nn.Module] = list(models)
        self.device = device
        self.device_policy = device_policy

        if device_policy == "all_on_device":
            for m in self.models:
                m.to(device)
                m.eval()
        elif device_policy == "swap_per_batch":
            for m in self.models:
                m.cpu()
                m.eval()
        elif device_policy == "cpu_only":
            for m in self.models:
                m.cpu()
                m.eval()
        else:
            raise ValueError(f"Unknown device_policy: {device_policy!r}")

    def __len__(self) -> int:
        return len(self.models)

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run ensemble inference and aggregate outputs.

        Parameters
        ----------
        X : torch.Tensor, shape (batch, 4, in_window)
            Input batch. Must already be on the correct device for the policy
            (``device`` for the GPU policies, CPU for ``cpu_only``).

        Returns
        -------
        profile_counts : torch.Tensor, shape (batch, n_outputs, out_window)
            Ensemble-averaged per-position counts (probs * totals, averaged
            across teachers).
        log_counts : torch.Tensor, shape (batch, 1)
            Log of summed totals (plus one for numerical stability), matching
            the original loop.
        """
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        track_accum = None

        for m in self.models:
            if self.device_policy == "swap_per_batch":
                m.to(self.device)

            logits, log_counts = m(X)
            flat = logits.reshape(logits.shape[0], -1)
            log_probs = log_softmax(flat).reshape_as(logits)
            probs = torch.exp(log_probs)
            total = torch.exp(log_counts) - 1.0
            track = probs * total.view(-1, 1, 1)

            if track_accum is None:
                track_accum = track.clone()
            else:
                track_accum = track_accum + track

            if self.device_policy == "swap_per_batch":
                m.cpu()
                torch.cuda.empty_cache()

            del logits, log_counts, flat, log_probs, probs, total, track

        assert track_accum is not None
        track_avg = track_accum / float(len(self.models))
        total_avg = (
            track_avg.reshape(track_avg.shape[0], -1).sum(dim=1, keepdim=True) + 1.0
        )
        log_counts_avg = torch.log(torch.clamp(total_avg, min=1e-12))
        return track_avg, log_counts_avg