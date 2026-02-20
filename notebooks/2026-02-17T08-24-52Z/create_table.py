from __future__ import annotations

from typing import Callable, Mapping

import nico2_lib as n2l
import numpy as np
import polars as pl

from experiment import ExperimentResult, ResultRecord


MetricFn = Callable[[np.ndarray, np.ndarray], float | np.ndarray]
_PREDICTION_SCOPES = ("global", "celltype")


def build_metric_registry() -> dict[str, MetricFn]:
    return {"pearsonr": n2l.mt.pearson_metric, "spearmanr": n2l.mt.spearman_metric}


def _score_celltype(metric: MetricFn, observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        np.nan_to_num(
            np.array(
                [
                    metric(observed_cell, predicted_cell)
                    for observed_cell, predicted_cell in zip(observed, predicted)
                ]
            )
        ).mean()
    )


def build_tidy_aggregated_scores_df(
    experiment_result: ExperimentResult,
    metric_registry: Mapping[str, MetricFn] | None = None,
) -> pl.DataFrame:
    metrics = build_metric_registry() if metric_registry is None else metric_registry
    rows: list[dict[str, object]] = []

    results_by_dataset_sample_model: dict[tuple[int, int, int], list[ResultRecord]] = {}
    for result_record in experiment_result.results:
        key = (result_record.dataset_id, result_record.sample_id, result_record.model_id)
        results_by_dataset_sample_model.setdefault(key, []).append(result_record)

    samples_by_dataset: dict[int, list[int]] = {}
    for sample in experiment_result.samples:
        samples_by_dataset.setdefault(sample.dataset_id, []).append(sample.sample_id)

    models_by_dataset: dict[int, list[tuple[int, str]]] = {}
    for model in experiment_result.models:
        models_by_dataset.setdefault(model.dataset_id, []).append(
            (model.model_id, model.model_name)
        )

    for dataset in experiment_result.datasets:
        sample_ids = samples_by_dataset.get(dataset.dataset_id, [])
        model_entries = models_by_dataset.get(dataset.dataset_id, [])
        for sample_id in sample_ids:
            for model_id, model_name in model_entries:
                grouped_records = results_by_dataset_sample_model.get(
                    (dataset.dataset_id, sample_id, model_id), []
                )
                if not grouped_records:
                    continue
                for prediction_scope in _PREDICTION_SCOPES:
                    for metric_name, metric in metrics.items():
                        celltype_scores: list[float] = []
                        celltype_weights: list[int] = []

                        for result_record in grouped_records:
                            observed = np.asarray(result_record.observed_test_counts)
                            predicted = np.asarray(
                                result_record.global_model_predicted_test_counts
                                if prediction_scope == "global"
                                else result_record.celltype_model_predicted_test_counts
                            )
                            celltype_scores.append(_score_celltype(metric, observed, predicted))
                            celltype_weights.append(result_record.n_cells)

                        if not celltype_scores:
                            continue

                        scores = np.asarray(celltype_scores, dtype=float)
                        weights = np.asarray(celltype_weights, dtype=float)
                        aggregations = {
                            "macro": float(scores.mean()),
                            "weighted": float(np.average(scores, weights=weights)),
                        }
                        for aggregation, score in aggregations.items():
                            rows.append(
                                {
                                    "dataset_name": dataset.dataset_name,
                                    "model_name": model_name,
                                    "prediction_scope": prediction_scope,
                                    "sample_idx": sample_id,
                                    "metric_name": metric_name,
                                    "aggregation": aggregation,
                                    "score": score,
                                    "n_celltypes": int(scores.size),
                                    "total_cells": int(weights.sum()),
                                }
                            )

    columns = [
        "dataset_name",
        "model_name",
        "prediction_scope",
        "sample_idx",
        "metric_name",
        "aggregation",
        "score",
        "n_celltypes",
        "total_cells",
    ]
    if not rows:
        return pl.DataFrame({column: [] for column in columns})

    return pl.DataFrame(rows).select(columns).sort(columns[:6])
