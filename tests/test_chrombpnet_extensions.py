import math

import torch

from bpnet_distill import (
    FineTuneConfig,
    VariantExample,
    chrombpnet_supervised_loss,
    effect_correlation,
    finetune_chrombpnet,
    gwas_enrichment_by_threshold,
    qtl_classification_metrics,
    score_variant_effects,
)


class TinyProfileModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 2, kernel_size=1)
        self.count = torch.nn.Linear(4, 1)

    def forward(self, X):
        profile_logits = self.conv(X)
        pooled = X.mean(dim=2)
        log_counts = self.count(pooled)
        return profile_logits, log_counts


def _one_hot(channel, length=6):
    X = torch.zeros(4, length)
    X[channel, :] = 1.0
    return X


def test_score_variant_effects_carries_metadata_and_scores():
    model = TinyProfileModel()
    variants = [
        VariantExample(
            "v1",
            ref_sequence=_one_hot(0),
            alt_sequence=_one_hot(1),
            label=1,
            observed_effect=0.5,
            pip=0.8,
            locus_id="locus-a",
        ),
        VariantExample("v2", _one_hot(2), _one_hot(3), label=0, pip=0.1),
    ]

    rows = score_variant_effects(
        model,
        variants,
        torch.device("cpu"),
        batch_size=1,
    )

    assert [row["variant_id"] for row in rows] == ["v1", "v2"]
    assert "log_count_delta" in rows[0]
    assert "profile_jsd" in rows[0]
    assert rows[0]["label"] == 1.0
    assert rows[0]["locus_id"] == "locus-a"


def test_qtl_and_gwas_benchmark_metrics():
    rows = [
        {"label": 1, "abs_log_count_delta": 0.9, "log_count_delta": 0.9, "observed_effect": 1.0, "pip": 0.9},
        {"label": 0, "abs_log_count_delta": 0.1, "log_count_delta": -0.1, "observed_effect": -0.2, "pip": 0.1},
        {"label": 1, "abs_log_count_delta": 0.8, "log_count_delta": 0.8, "observed_effect": 0.7, "pip": 0.8},
        {"label": 0, "abs_log_count_delta": 0.2, "log_count_delta": -0.2, "observed_effect": -0.3, "pip": 0.2},
    ]

    qtl = qtl_classification_metrics(rows)
    corr = effect_correlation(rows)
    enrich = gwas_enrichment_by_threshold(
        rows,
        score_thresholds=(0.5,),
        pip_thresholds=(0.5,),
    )

    assert qtl["average_precision"] == 1.0
    assert qtl["roc_auc"] == 1.0
    assert corr["pearson"] > 0.99
    assert enrich[0]["overlap"] == 2.0
    assert enrich[0]["enrichment"] == 2.0


def test_supervised_loss_and_finetuning_loop_run():
    model = TinyProfileModel()
    X = torch.stack([_one_hot(0), _one_hot(1)])
    y = torch.ones(2, 2, 6)
    outputs = model(X)

    loss = chrombpnet_supervised_loss(outputs, y)
    assert torch.isfinite(loss)

    loader = [(X, y)]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    metrics = finetune_chrombpnet(
        model,
        loader,
        None,
        optimizer,
        torch.device("cpu"),
        config=FineTuneConfig(max_epochs=1, verbose=False),
    )

    assert len(metrics) == 1
    assert math.isfinite(metrics[0]["train_loss"])
