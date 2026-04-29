"""SPLADE sparse-learned first-stage retriever.

Cell-equivalent: SPLADE (cell 38, execution trimmed).
"""

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, SPLADE_MODEL_NAME, TOP_K
from src.data_loader import (
    derive_unique_job_queries, derive_unique_resume_candidates,
    build_ranking_results,
)
from src.stage_1_tfidf import compose_job_text, compose_resume_text
from eval.evaluator import evaluate_rankings, save_metric_bundle


# SPLADE ranking pipeline and full-test run.

try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    SPLADE_AVAILABLE = True
except ImportError:
    SPLADE_AVAILABLE = False
    print("transformers not installed. Install with: pip install transformers")

import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load .env file
load_dotenv()

_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

if _HF_TOKEN:
    login(_HF_TOKEN)

class SPLADEVectorizer:
    """Wraps a SPLADE model to produce sparse term-weight vectors."""

    def __init__(self, model_name: str = SPLADE_MODEL_NAME, max_length: int = 256, batch_size: int = 32, device: str | None = None):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str]) -> tuple[np.ndarray, dict]:
        """Encode texts into SPLADE sparse vectors. Returns (sparse_matrix, vocab_map)."""
        import torch
        all_weights = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            tokens = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.no_grad():
                output = self.model(**tokens)
            logits = output.logits  # (batch, seq_len, vocab_size)
            # SPLADE: max-pool over tokens, then ReLU + log(1+x)
            sparse_weights = torch.max(
                torch.log1p(torch.relu(logits)) * tokens["attention_mask"].unsqueeze(-1),
                dim=1,
            ).values  # (batch, vocab_size)
            all_weights.append(sparse_weights.cpu().numpy())
        weight_matrix = np.vstack(all_weights)  # (n_texts, vocab_size)
        vocab = {i: token for i, token in enumerate(self.tokenizer.get_vocab()) if i < weight_matrix.shape[1]}
        return weight_matrix, vocab


def rank_with_splade(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    vectorizer: SPLADEVectorizer | None = None,
    model_name: str = "splade",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    query_table = queries.copy().reset_index(drop=True)
    candidate_table = candidates.copy().reset_index(drop=True)
    query_table["query_text"] = compose_job_text(query_table)
    candidate_table["candidate_text"] = compose_resume_text(candidate_table)

    if vectorizer is None:
        vectorizer = SPLADEVectorizer()

    query_weights, _ = vectorizer.encode(query_table["query_text"].tolist())
    candidate_weights, _ = vectorizer.encode(candidate_table["candidate_text"].tolist())

    # Dot-product similarity (sparse vectors are dense numpy arrays here)
    score_matrix = query_weights @ candidate_weights.T

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
        "query_weights": query_weights,
        "candidate_weights": candidate_weights,
    }
    return rankings, context


def run_splade_experiment(split_frame: pd.DataFrame, artifact_prefix: str = "splade_full_test_metrics") -> dict[str, Any]:
    queries = derive_unique_job_queries(split_frame)
    candidates = derive_unique_resume_candidates(split_frame)
    start = time.perf_counter()
    vectorizer = SPLADEVectorizer()
    rankings, context = rank_with_splade(queries, candidates, vectorizer, model_name="splade")
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
