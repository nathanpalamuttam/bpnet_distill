"""Variant-effect benchmarks for ChromBPNet/BPNet-style models.

These helpers cover the benchmark families emphasized in the ChromBPNet paper:

- allelic count effects and profile divergence for variants,
- QTL classification and effect-size correlation,
- enrichment of high-effect predictions among fine-mapped GWAS variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np
import torch


@dataclass
class VariantExample:
    """A ref/alt sequence pair with optional benchmark annotations."""

    variant_id: str
    ref_sequence: torch.Tensor
    alt_sequence: torch.Tensor
    label: Optional[int] = None
    observed_effect: Optional[float] = None
    pip: Optional[float] = None
    locus_id: Optional[str] = None


def _as_sequence_tensor(sequence) -> torch.Tensor:
    tensor = torch.as_tensor(sequence, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[0] != 4:
        raise ValueError(
            "Sequences must be one-hot tensors or arrays with shape (4, length)."
        )
    return tensor


def _batched(
    items: Sequence[VariantExample],
    batch_size: int,
) -> Iterator[Sequence[VariantExample]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _profile_probs(profile_logits: torch.Tensor) -> torch.Tensor:
    flat = profile_logits.reshape(profile_logits.shape[0], -1)
    probs = torch.nn.functional.softmax(flat, dim=-1)
    return probs.reshape_as(profile_logits)


def _jensen_shannon(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.reshape(p.shape[0], -1).clamp_min(eps)
    q = q.reshape(q.shape[0], -1).clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    m = 0.5 * (p + q)
    return 0.5 * (
        torch.sum(p * torch.log(p / m), dim=1)
        + torch.sum(q * torch.log(q / m), dim=1)
    )


@torch.no_grad()
def score_variant_effects(
    model: torch.nn.Module,
    variants: Iterable[VariantExample],
    device: torch.device,
    *,
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    """Score ref/alt sequence pairs with count and profile effects.

    Returns one dictionary per variant with ``log_count_delta`` (alt - ref)
    and ``profile_jsd``. Optional labels, observed effects, PIPs, and locus IDs
    are carried through for downstream benchmark functions.
    """
    items = list(variants)
    model.to(device)
    model.eval()
    rows: List[Dict[str, Any]] = []

    for batch in _batched(items, batch_size):
        ref = torch.stack([_as_sequence_tensor(v.ref_sequence) for v in batch]).to(
            device
        )
        alt = torch.stack([_as_sequence_tensor(v.alt_sequence) for v in batch]).to(
            device
        )

        ref_profile_logits, ref_log_counts = model(ref)
        alt_profile_logits, alt_log_counts = model(alt)

        ref_log_counts = ref_log_counts.reshape(ref_log_counts.shape[0], -1).sum(dim=1)
        alt_log_counts = alt_log_counts.reshape(alt_log_counts.shape[0], -1).sum(dim=1)
        jsd = _jensen_shannon(
            _profile_probs(ref_profile_logits),
            _profile_probs(alt_profile_logits),
        )

        for i, variant in enumerate(batch):
            row: Dict[str, Any] = {
                "variant_id": variant.variant_id,
                "ref_log_count": float(ref_log_counts[i].detach().cpu()),
                "alt_log_count": float(alt_log_counts[i].detach().cpu()),
                "log_count_delta": float(
                    (alt_log_counts[i] - ref_log_counts[i]).detach().cpu()
                ),
                "abs_log_count_delta": float(
                    torch.abs(alt_log_counts[i] - ref_log_counts[i]).detach().cpu()
                ),
                "profile_jsd": float(jsd[i].detach().cpu()),
            }
            if variant.label is not None:
                row["label"] = float(variant.label)
            if variant.observed_effect is not None:
                row["observed_effect"] = float(variant.observed_effect)
            if variant.pip is not None:
                row["pip"] = float(variant.pip)
            if variant.locus_id is not None:
                row["locus_id"] = variant.locus_id
            rows.append(row)

    return rows


def average_precision_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute average precision without requiring scikit-learn."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / (np.arange(y_sorted.size) + 1)
    return float(precision[y_sorted].sum() / y.sum())


def roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute ROC AUC from rank statistics without requiring scikit-learn."""
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)

    unique_scores, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    _ = unique_scores
    for group in np.where(counts > 1)[0]:
        tied = inverse == group
        ranks[tied] = ranks[tied].mean()

    rank_sum_pos = ranks[y].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def qtl_classification_metrics(
    rows: Sequence[Mapping[str, float]],
    *,
    score_key: str = "abs_log_count_delta",
    label_key: str = "label",
) -> Dict[str, float]:
    """Benchmark predicted variant effects against binary QTL labels."""
    labels = [
        int(row[label_key])
        for row in rows
        if label_key in row and score_key in row
    ]
    scores = [
        float(row[score_key])
        for row in rows
        if label_key in row and score_key in row
    ]
    return {
        "average_precision": average_precision_score(labels, scores),
        "roc_auc": roc_auc_score(labels, scores),
        "n_variants": float(len(labels)),
        "n_positive": float(sum(labels)),
    }


def effect_correlation(
    rows: Sequence[Mapping[str, float]],
    *,
    predicted_key: str = "log_count_delta",
    observed_key: str = "observed_effect",
) -> Dict[str, float]:
    """Compute signed and unsigned Pearson correlations for QTL effects."""
    pairs = [
        (float(row[predicted_key]), float(row[observed_key]))
        for row in rows
        if predicted_key in row and observed_key in row
    ]
    if len(pairs) < 2:
        return {
            "pearson": float("nan"),
            "abs_pearson": float("nan"),
            "n_variants": float(len(pairs)),
        }
    pred = np.asarray([p[0] for p in pairs], dtype=float)
    obs = np.asarray([p[1] for p in pairs], dtype=float)
    return {
        "pearson": _pearson(pred, obs),
        "abs_pearson": _pearson(np.abs(pred), np.abs(obs)),
        "n_variants": float(len(pairs)),
    }


def gwas_enrichment_by_threshold(
    rows: Sequence[Mapping[str, float]],
    *,
    score_key: str = "abs_log_count_delta",
    pip_key: str = "pip",
    score_thresholds: Sequence[float] = (0.25, 0.5, 1.0),
    pip_thresholds: Sequence[float] = (0.1, 0.5, 0.9),
) -> List[Dict[str, float]]:
    """Compute overlap enrichment of high-effect calls with fine-mapped SNPs."""
    filtered = [row for row in rows if score_key in row and pip_key in row]
    if not filtered:
        return []

    scores = np.asarray([abs(float(row[score_key])) for row in filtered], dtype=float)
    pips = np.asarray([float(row[pip_key]) for row in filtered], dtype=float)
    n = float(scores.size)
    results: List[Dict[str, float]] = []

    for score_threshold in score_thresholds:
        high_score = scores >= score_threshold
        for pip_threshold in pip_thresholds:
            high_pip = pips >= pip_threshold
            overlap = float(np.logical_and(high_score, high_pip).sum())
            expected = float(high_score.sum() * high_pip.sum() / n)
            enrichment = overlap / expected if expected > 0 else float("nan")
            results.append(
                {
                    "score_threshold": float(score_threshold),
                    "pip_threshold": float(pip_threshold),
                    "n_variants": n,
                    "n_high_score": float(high_score.sum()),
                    "n_high_pip": float(high_pip.sum()),
                    "overlap": overlap,
                    "expected_overlap": expected,
                    "enrichment": enrichment,
                }
            )
    return results


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return float("nan")
    return float(np.sum(x * y) / denom)
