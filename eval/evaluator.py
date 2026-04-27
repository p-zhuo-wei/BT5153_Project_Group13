"""Per-query + aggregate evaluation, save_metric_bundle, paired bootstrap.

Cell-equivalent: Step 2C (cell 25).
"""

import json
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, RANDOM_SEED
from src.data_loader import attach_relevance_labels, build_relevance_lookup
from eval.metrics import (
    precision_at_k, recall_at_k, reciprocal_rank,
    average_precision, ndcg_at_k,
)


# Aggregate evaluation and bootstrap significance helpers.

def evaluate_rankings(rankings: pd.DataFrame, split_frame: pd.DataFrame, ks: Sequence[int] = (1, 3, 5, 10)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rankings = attach_relevance_labels(rankings, split_frame)
    _, relevance_lookup = build_relevance_lookup(split_frame)
    rows = []
    for (model_name, split_name), model_group in rankings.groupby(["model_name", "split"], sort=True):
        split_queries = sorted(split_frame.loc[split_frame["split"] == split_name, "job_id"].drop_duplicates().tolist())
        for job_id in split_queries:
            group = model_group.loc[model_group["job_id"] == job_id].sort_values("rank")
            ranked_relevances = group["relevance_label"].tolist()
            ideal_relevances = sorted(relevance_lookup.get(job_id, {}).values(), reverse=True)
            total_relevant = sum(1 for value in ideal_relevances if value >= 1)
            row = {
                "model_name": model_name,
                "split": split_name,
                "job_id": job_id,
                "retrieved_count": len(ranked_relevances),
                "relevant_count": total_relevant,
                "mrr": reciprocal_rank(ranked_relevances),
                "map": average_precision(ranked_relevances, total_relevant=total_relevant),
            }
            for k in ks:
                row[f"precision@{k}"] = precision_at_k(ranked_relevances, k)
                row[f"recall@{k}"] = recall_at_k(ranked_relevances, k, total_relevant=total_relevant)
                row[f"ndcg@{k}"] = ndcg_at_k(ranked_relevances, k, ideal_relevances=ideal_relevances)
            rows.append(row)
    query_metrics = pd.DataFrame(rows).sort_values(["model_name", "split", "job_id"]).reset_index(drop=True)

    aggregate_rows = []
    for (model_name, split_name), group in query_metrics.groupby(["model_name", "split"], sort=True):
        aggregate = {
            "model_name": model_name,
            "split": split_name,
            "query_count": int(len(group)),
            "mrr": float(group["mrr"].mean()),
            "map": float(group["map"].mean()),
        }
        for k in ks:
            aggregate[f"precision@{k}"] = float(group[f"precision@{k}"].mean())
            aggregate[f"recall@{k}"] = float(group[f"recall@{k}"].mean())
            aggregate[f"ndcg@{k}"] = float(group[f"ndcg@{k}"].mean())
        aggregate_rows.append(aggregate)
    aggregate_metrics = pd.DataFrame(aggregate_rows)
    return aggregate_metrics, query_metrics


def paired_bootstrap_significance_test(left_query_metrics: pd.DataFrame, right_query_metrics: pd.DataFrame, metrics: Sequence[str], n_bootstrap: int = 5000, confidence_level: float = 0.95) -> pd.DataFrame:
    merged = left_query_metrics[["job_id", *metrics]].merge(
        right_query_metrics[["job_id", *metrics]],
        on="job_id",
        how="inner",
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        raise ValueError("No shared jobs found for paired bootstrap testing.")

    rng = np.random.default_rng(RANDOM_SEED)
    alpha = 1.0 - confidence_level
    rows = []
    for metric in metrics:
        deltas = merged[f"{metric}_left"].to_numpy(dtype=float) - merged[f"{metric}_right"].to_numpy(dtype=float)
        sample_count = len(deltas)
        sample_indices = rng.integers(0, sample_count, size=(n_bootstrap, sample_count))
        bootstrap_distribution = deltas[sample_indices].mean(axis=1)
        rows.append(
            {
                "metric": metric,
                "query_count": int(sample_count),
                "observed_delta": float(deltas.mean()),
                "ci_lower": float(np.quantile(bootstrap_distribution, alpha / 2.0)),
                "ci_upper": float(np.quantile(bootstrap_distribution, 1.0 - alpha / 2.0)),
                "p_value": float(min(2.0 * min(np.mean(bootstrap_distribution <= 0.0), np.mean(bootstrap_distribution >= 0.0)), 1.0)),
            }
        )
    return pd.DataFrame(rows)


def save_metric_bundle(prefix: str, aggregate_metrics: pd.DataFrame, query_metrics: pd.DataFrame, rankings: pd.DataFrame | None = None) -> None:
    aggregate_metrics.to_csv(METRICS_DIR / f"{prefix}.csv", index=False)
    query_metrics.to_csv(METRICS_DIR / f"{prefix.replace('_metrics', '')}_query_metrics.csv", index=False)
    (METRICS_DIR / f"{prefix}.json").write_text(json.dumps(aggregate_metrics.to_dict(orient="records"), indent=2))
    if rankings is not None:
        rankings.to_csv(METRICS_DIR / f"{prefix.replace('_metrics', '')}_rankings.csv", index=False)
