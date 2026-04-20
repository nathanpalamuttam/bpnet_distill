"""Default loss functions for profile+count distillation.

Matches the bpnet-lite convention: MNLL on the profile head, MSE on the
log-count head, combined as ``total = mnll + alpha * mse``.

The user-facing trainer accepts these separately and a scalar ``alpha``
weight. The combined-loss-callable override lives as a TODO below.

TODO
----
- Accept a single unified ``loss_fn(pred_profile, pred_counts, target_profile,
  target_counts, mask) -> scalar`` as an alternative to the separate
  profile/count callables. Would require threading the combined loss through
  ``Model.fit_generator`` (which currently computes MNLL + MSE internally).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mnll_loss(
    logps: torch.Tensor,
    true_counts: torch.Tensor,
) -> torch.Tensor:
    """Multinomial negative log-likelihood on profile predictions.

    Parameters
    ----------
    logps : torch.Tensor, shape (batch, n_outputs * out_window)
        Log-probabilities over positions (log-softmax output), flattened per
        example across strands and positions.
    true_counts : torch.Tensor, shape (batch, n_outputs * out_window)
        Observed counts per position, flattened to match ``logps``.

    Returns
    -------
    torch.Tensor, shape (batch,)
        Per-example negative log-likelihoods.
    """
    total_counts = true_counts.sum(dim=-1)
    # lgamma(n+1) - sum lgamma(x+1) - sum x * logp
    log_fact_sum = torch.lgamma(total_counts + 1)
    log_prod_fact = torch.lgamma(true_counts + 1).sum(dim=-1)
    log_prob = (true_counts * logps).sum(dim=-1)
    return -(log_fact_sum - log_prod_fact + log_prob)


def log_count_mse_loss(
    pred_log_counts: torch.Tensor,
    true_counts: torch.Tensor,
) -> torch.Tensor:
    """MSE on log(1 + counts).

    Parameters
    ----------
    pred_log_counts : torch.Tensor, shape (batch, 1) or (batch, n_outputs)
        Predicted log-counts from the count head.
    true_counts : torch.Tensor, shape (batch, n_outputs, out_window) or (batch, L)
        Observed counts. Summed across non-batch dims before comparison.

    Returns
    -------
    torch.Tensor, scalar
        Mean squared error between ``pred_log_counts`` and
        ``log(1 + true_counts.sum())``.
    """
    target = torch.log(1 + true_counts.reshape(true_counts.shape[0], -1).sum(dim=-1))
    return F.mse_loss(pred_log_counts.squeeze(-1), target)