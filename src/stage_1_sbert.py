"""SBERT dense first-stage retriever: embedding cache, matcher, runner.

Cell-equivalents: Step 4A (cell 34) + Step 4B (cell 36, execution trimmed).
"""

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDINGS_DIR, METRICS_DIR, SBERT_MODEL_NAME, TOP_K
from src.data_loader import (
    derive_unique_job_queries, derive_unique_resume_candidates,
    build_ranking_results,
)
from src.stage_1_tfidf import compose_job_text, compose_resume_text
from eval.evaluator import evaluate_rankings, save_metric_bundle


# SBERT wrapper and embedding-cache helpers.

def corpus_hash(frame: pd.DataFrame, id_col: str, text_col: str) -> str:
    digest = []
    normalized = frame[[id_col, text_col]].copy()
    normalized[text_col] = normalized[text_col].fillna("").astype(str)
    normalized = normalized.drop_duplicates(subset=[id_col]).sort_values(id_col, kind="mergesort")
    for row in normalized.itertuples(index=False):
        digest.append(f"{getattr(row, id_col)}\x1f{getattr(row, text_col)}")
    return str(abs(hash("\x1e".join(digest))))[:16]


class NotebookSBERTMatcher:
    def __init__(self, model_name: str = SBERT_MODEL_NAME, batch_size: int = 64, device: str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        kwargs = {"device": device} if device is not None else {}
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode_unique_texts(self, frame: pd.DataFrame, split: str, entity: str, id_col: str, text_col: str, use_cache: bool = True) -> tuple[np.ndarray, np.ndarray, Path | None]:
        normalized = frame[[id_col, text_col]].copy()
        normalized[text_col] = normalized[text_col].fillna("").astype(str)
        normalized = normalized.drop_duplicates(subset=[id_col]).reset_index(drop=True)
        ids = normalized[id_col].to_numpy()
        texts = normalized[text_col].tolist()
        cache_name = f"sbert_{entity}_{split}_{corpus_hash(normalized, id_col, text_col)}.npz"
        cache_path = EMBEDDINGS_DIR / cache_name
        if use_cache and cache_path.exists():
            cached = np.load(cache_path)
            return cached["ids"], cached["embeddings"].astype(np.float32, copy=False), cache_path
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        if use_cache:
            np.savez_compressed(cache_path, ids=ids, embeddings=embeddings)
        return ids, embeddings, cache_path if use_cache else None

# SBERT ranking pipeline and full-test run.

def rank_with_sbert(queries: pd.DataFrame, candidates: pd.DataFrame, matcher: NotebookSBERTMatcher, model_name: str = "sbert") -> tuple[pd.DataFrame, dict[str, Any]]:
    query_table = queries.copy().reset_index(drop=True)
    candidate_table = candidates.copy().reset_index(drop=True)
    job_ids, job_embeddings, job_cache = matcher.encode_unique_texts(query_table, str(query_table["split"].iloc[0]), "jobs", "job_id", "job_text")
    resume_ids, resume_embeddings, resume_cache = matcher.encode_unique_texts(candidate_table, str(candidate_table["split"].iloc[0]), "resumes", "resume_id", "resume_text")
    score_matrix = job_embeddings @ resume_embeddings.T
    scored_pairs = pd.DataFrame({
        "job_id": np.repeat(job_ids, score_matrix.shape[1]),
        "resume_id": resume_ids[np.argsort(-score_matrix, axis=1)].reshape(-1),
        "score": np.take_along_axis(score_matrix, np.argsort(-score_matrix, axis=1), axis=1).reshape(-1),
    })
    rankings = build_ranking_results(scored_pairs, model_name=model_name, split=str(query_table["split"].iloc[0]))
    context = {
        "job_ids": job_ids,
        "resume_ids": resume_ids,
        "job_embeddings": job_embeddings,
        "resume_embeddings": resume_embeddings,
        "job_cache": job_cache,
        "resume_cache": resume_cache,
        "queries": query_table,
        "candidates": candidate_table,
        "matcher": matcher,
    }
    return rankings, context


def run_sbert_experiment(split_frame: pd.DataFrame, artifact_prefix: str = "sbert_full_test_metrics") -> dict[str, Any]:
    queries = derive_unique_job_queries(split_frame)
    candidates = derive_unique_resume_candidates(split_frame)
    start = time.perf_counter()
    matcher = NotebookSBERTMatcher()
    rankings, context = rank_with_sbert(queries, candidates, matcher, model_name="sbert")
    runtime_seconds = time.perf_counter() - start
    aggregate_metrics, query_metrics = evaluate_rankings(rankings, split_frame)
    save_metric_bundle(artifact_prefix, aggregate_metrics, query_metrics, rankings=rankings)
    return {
        "rankings": rankings,
        "aggregate_metrics": aggregate_metrics,
        "query_metrics": query_metrics,
        "context": context,
        "runtime_seconds": runtime_seconds,
    }
