"""TF-IDF first-stage retriever: text views, ranker, full-test runner.

Cell-equivalents: Step 3A (cell 28) + Step 3B (cell 30, execution trimmed).
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import METRICS_DIR, TABLES_DIR, TOP_K
from src.data_loader import (
    derive_unique_job_queries, derive_unique_resume_candidates,
    attach_relevance_labels, build_ranking_results,
)
from eval.evaluator import evaluate_rankings, save_metric_bundle


# TF-IDF input views and lexical explanation helper.

def compose_resume_text(frame: pd.DataFrame) -> pd.Series:
    pieces = [
        frame.get("resume_category", pd.Series("", index=frame.index)).fillna("").astype(str),
        frame.get("resume_text", pd.Series("", index=frame.index)).fillna("").astype(str),
    ]
    return pd.Series([" ".join(part for part in parts if part).strip() for parts in zip(*pieces, strict=False)], index=frame.index)


def compose_job_text(frame: pd.DataFrame) -> pd.Series:
    pieces = [
        frame.get("job_title", pd.Series("", index=frame.index)).fillna("").astype(str),
        frame.get("job_function", pd.Series("", index=frame.index)).fillna("").astype(str),
        frame.get("job_industry", pd.Series("", index=frame.index)).fillna("").astype(str),
        frame.get("job_category", pd.Series("", index=frame.index)).fillna("").astype(str),
        frame.get("job_text", pd.Series("", index=frame.index)).fillna("").astype(str),
    ]
    return pd.Series([" ".join(part for part in parts if part).strip() for parts in zip(*pieces, strict=False)], index=frame.index)


def explain_tfidf_match(job_id: Any, resume_id: Any, queries: pd.DataFrame, candidates: pd.DataFrame, vectorizer: TfidfVectorizer, query_matrix, candidate_matrix, top_n: int = 8) -> list[dict[str, float]]:
    feature_names = vectorizer.get_feature_names_out()
    query_idx = queries.index[queries["job_id"] == job_id][0]
    candidate_idx = candidates.index[candidates["resume_id"] == resume_id][0]
    query_vector = query_matrix.getrow(query_idx)
    candidate_vector = candidate_matrix.getrow(candidate_idx)
    overlap = query_vector.multiply(candidate_vector).tocoo()
    if overlap.nnz == 0:
        return []
    order = np.argsort(overlap.data)[::-1][:top_n]
    return [
        {
            "term": str(feature_names[int(overlap.col[idx])]),
            "contribution": float(overlap.data[idx]),
            "query_weight": float(query_vector[0, overlap.col[idx]]),
            "candidate_weight": float(candidate_vector[0, overlap.col[idx]]),
        }
        for idx in order
    ]

# TF-IDF ranking pipeline and full-test run.

def rank_with_tfidf(queries: pd.DataFrame, candidates: pd.DataFrame, model_name: str = "tfidf") -> tuple[pd.DataFrame, dict[str, Any]]:
    query_table = queries.copy().reset_index(drop=True)
    candidate_table = candidates.copy().reset_index(drop=True)
    query_table["query_text"] = compose_job_text(query_table)
    candidate_table["candidate_text"] = compose_resume_text(candidate_table)

    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    corpus = pd.concat([query_table["query_text"], candidate_table["candidate_text"]], axis=0, ignore_index=True)
    vectorizer.fit(corpus.tolist())
    query_matrix = vectorizer.transform(query_table["query_text"].tolist()).tocsr()
    candidate_matrix = vectorizer.transform(candidate_table["candidate_text"].tolist()).tocsr()
    score_matrix = (query_matrix @ candidate_matrix.T).toarray()

    rows = []
    for row_index, job_id in enumerate(query_table["job_id"].to_numpy()):
        rows.append(pd.DataFrame({
            "job_id": job_id,
            "resume_id": candidate_table["resume_id"].to_numpy(),
            "score": score_matrix[row_index].astype(float, copy=False),
        }))
    scored_pairs = pd.concat(rows, ignore_index=True)
    rankings = build_ranking_results(scored_pairs, model_name=model_name, split=str(query_table["split"].iloc[0]))
    context = {
        "queries": query_table,
        "candidates": candidate_table,
        "vectorizer": vectorizer,
        "query_matrix": query_matrix,
        "candidate_matrix": candidate_matrix,
    }
    return rankings, context


def run_tfidf_experiment(split_frame: pd.DataFrame, artifact_prefix: str = "tfidf_full_test_metrics") -> dict[str, Any]:
    queries = derive_unique_job_queries(split_frame)
    candidates = derive_unique_resume_candidates(split_frame)
    start = time.perf_counter()
    rankings, context = rank_with_tfidf(queries, candidates, model_name="tfidf")
    runtime_seconds = time.perf_counter() - start
    aggregate_metrics, query_metrics = evaluate_rankings(rankings, split_frame)
    save_metric_bundle(artifact_prefix, aggregate_metrics, query_metrics, rankings=rankings)

    top_rows = attach_relevance_labels(rankings, split_frame)
    top_rows = top_rows.loc[top_rows["rank"] == 1].sort_values(["score", "job_id"], ascending=[False, True]).head(12)
    explanations = []
    for row in top_rows.itertuples(index=False):
        top_terms = explain_tfidf_match(row.job_id, row.resume_id, context["queries"], context["candidates"], context["vectorizer"], context["query_matrix"], context["candidate_matrix"])
        for position, item in enumerate(top_terms, start=1):
            explanations.append({
                "job_id": row.job_id,
                "resume_id": row.resume_id,
                "score": row.score,
                "term_rank": position,
                **item,
            })
    explanation_frame = pd.DataFrame(explanations)
    explanation_frame.to_csv(TABLES_DIR / "tfidf_top_match_terms.csv", index=False)
    return {
        "rankings": rankings,
        "aggregate_metrics": aggregate_metrics,
        "query_metrics": query_metrics,
        "context": context,
        "runtime_seconds": runtime_seconds,
        "top_terms": explanation_frame,
    }
