"""Cross-encoder second-stage reranker (generic over first-stage shortlists).

Cell-equivalent: Cross-encoder (cell 44, execution trimmed).
"""

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.config import (
    CROSS_ENCODER_MODEL_NAME, METRICS_DIR, RERANK_SHORTLIST_K, TOP_K,
)
from src.data_loader import (
    derive_unique_job_queries, derive_unique_resume_candidates,
)
from src.stage_1_tfidf import compose_job_text, compose_resume_text
from src.data_loader import build_ranking_results
from eval.evaluator import evaluate_rankings, save_metric_bundle


# Cross-encoder reranking pipeline — generic: accepts any first-stage shortlist.

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
    # MiniLM-L-6 is small enough that MPS host<->device transfer overhead and
    # PyTorch 2.7 MPS memory bugs make CPU the more reliable choice.
    CROSS_ENCODER_DEVICE = "cpu"
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CROSS_ENCODER_DEVICE = "cpu"
    print("CrossEncoder not available. Requires sentence-transformers >= 2.0")

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_with_cross_encoder(
    first_stage_rankings: pd.DataFrame,
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    shortlist_k: int = RERANK_SHORTLIST_K,
    output_k: int = TOP_K,
    model_name: str = "cross_encoder",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rerank top-K from any first-stage rankings using a cross-encoder.

    Args:
        first_stage_rankings: Rankings DataFrame with columns [model_name, split, job_id, rank, resume_id, score].
        queries: DataFrame with job text information.
        candidates: DataFrame with resume text information.
        shortlist_k: How many top candidates per query to rerank.
        output_k: How many top candidates to return after reranking.
        model_name: Label for the output rankings.
    """
    query_table = queries.copy().reset_index(drop=True)
    candidate_table = candidates.copy().reset_index(drop=True)

    # Build text lookups
    query_table["query_text"] = compose_job_text(query_table)
    candidate_table["candidate_text"] = compose_resume_text(candidate_table)
    job_text_lookup = dict(zip(query_table["job_id"].to_numpy(), query_table["query_text"].fillna("").astype(str)))
    resume_text_lookup = dict(zip(candidate_table["resume_id"].to_numpy(), candidate_table["candidate_text"].fillna("").astype(str)))

    cross_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=CROSS_ENCODER_DEVICE)

    rows = []
    unique_job_ids = first_stage_rankings["job_id"].unique()
    batch_size = 64
    # CrossEncoder.predict batches per call (per query), not across calls,
    # so the total is queries * batches-per-query, not pairs / batch_size.
    total_batches_estimate = len(unique_job_ids) * math.ceil(shortlist_k / batch_size)
    batches_done = 0
    progress = tqdm(
        unique_job_ids,
        total=len(unique_job_ids),
        desc=f"CE rerank ({model_name})",
        unit="query",
    )
    for job_id in progress:
        job_shortlist = first_stage_rankings.loc[
            first_stage_rankings["job_id"] == job_id
        ].sort_values("rank").head(shortlist_k)

        pairs = []
        pair_resume_ids = []
        for _, row in job_shortlist.iterrows():
            rid = row["resume_id"]
            pairs.append([job_text_lookup.get(job_id, ""), resume_text_lookup.get(rid, "")])
            pair_resume_ids.append(rid)

        if not pairs:
            continue

        ce_scores = cross_model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        batches_done += math.ceil(len(pairs) / batch_size)
        progress.set_postfix(batches=f"{batches_done}/{total_batches_estimate}")
        ce_sorted = np.argsort(-ce_scores)[:output_k]
        for rank_idx in ce_sorted:
            rows.append({
                "job_id": job_id,
                "resume_id": pair_resume_ids[rank_idx],
                "score": float(ce_scores[rank_idx]),
            })

    scored_pairs = pd.DataFrame(rows)
    rankings = build_ranking_results(scored_pairs, model_name=model_name, split=str(query_table["split"].iloc[0]))
    context = {
        "queries": query_table,
        "candidates": candidate_table,
        "cross_encoder_model": CROSS_ENCODER_MODEL_NAME,
        "shortlist_k": shortlist_k,
    }
    return rankings, context


def run_cross_encoder_on_first_stage(
    first_stage_outputs: dict[str, Any],
    split_frame: pd.DataFrame,
    first_stage_label: str,
    artifact_prefix: str | None = None,
) -> dict[str, Any]:
    """Run cross-encoder reranking on any first-stage model's output."""
    queries = derive_unique_job_queries(split_frame)
    candidates = derive_unique_resume_candidates(split_frame)
    model_label = f"ce_rerank_{first_stage_label}"
    start = time.perf_counter()
    rankings, context = rerank_with_cross_encoder(
        first_stage_outputs["rankings"], queries, candidates,
        model_name=model_label,
    )
    runtime_seconds = time.perf_counter() - start
    aggregate_metrics, query_metrics = evaluate_rankings(rankings, split_frame)
    if artifact_prefix:
        save_metric_bundle(artifact_prefix, aggregate_metrics, query_metrics)
    return {
        "rankings": rankings,
        "aggregate_metrics": aggregate_metrics,
        "query_metrics": query_metrics,
        "context": context,
        "runtime_seconds": runtime_seconds,
    }
