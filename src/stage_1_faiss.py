"""FAISS approximate nearest-neighbor first-stage retriever.

Cell-equivalents: FAISS (cells 40, 41 — execution trimmed).
"""

import os
# Prevent threading conflict between SBERT (PyTorch) and FAISS.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import METRICS_DIR, TOP_K
from src.data_loader import (
    derive_unique_job_queries, derive_unique_resume_candidates,
    build_ranking_results,
)
from src.stage_1_sbert import NotebookSBERTMatcher
from src.stage_1_tfidf import compose_job_text, compose_resume_text
from eval.evaluator import evaluate_rankings, save_metric_bundle




import time
import numpy as np
import pandas as pd
from typing import Any

# Ensure faiss is installed: pip install faiss-cpu
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("faiss-cpu not installed. Install with: pip install faiss-cpu")

def rank_with_faiss(queries: pd.DataFrame, candidates: pd.DataFrame, matcher: NotebookSBERTMatcher, model_name: str = "faiss") -> tuple[pd.DataFrame, dict[str, Any]]:
    query_table = queries.copy().reset_index(drop=True)
    candidate_table = candidates.copy().reset_index(drop=True)
    
    # Reusing your existing SBERT encoding logic exactly as written
    job_ids, job_embeddings, job_cache = matcher.encode_unique_texts(query_table, str(query_table["split"].iloc[0]), "jobs", "job_id", "job_text")
    resume_ids, resume_embeddings, resume_cache = matcher.encode_unique_texts(candidate_table, str(candidate_table["split"].iloc[0]), "resumes", "resume_id", "resume_text")

    # --- CRITICAL SAFETY STEPS FOR FAISS ---
    # 1. Ensure memory is contiguous (prevents kernel crash)
    # 2. Ensure data type is float32 (prevents "nvn stop" / infinite loop)
    res_data = np.ascontiguousarray(resume_embeddings.astype(np.float32))
    qry_data = np.ascontiguousarray(job_embeddings.astype(np.float32))

    dim = res_data.shape[1]
    index = faiss.IndexFlatIP(dim)
    
    print("Adding vectors to FAISS...")
    index.add(res_data)

    print("Searching...")
    top_k_val = min(len(resume_ids), TOP_K)
    scores, indices = index.search(qry_data, top_k_val)
    print("Search complete.")
    # ---------------------------------------

    rows = []
    for q_idx in range(len(job_ids)):
        for rank_pos in range(scores.shape[1]):
            r_idx = indices[q_idx, rank_pos]
            if r_idx < 0: continue
            rows.append({
                "job_id": job_ids[q_idx],
                "resume_id": resume_ids[r_idx],
                "score": float(scores[q_idx, rank_pos]),
            })
            
    scored_pairs = pd.DataFrame(rows)
    rankings = build_ranking_results(scored_pairs, model_name=model_name, split=str(query_table["split"].iloc[0]))
    
    context = {
        "job_ids": job_ids, "resume_ids": resume_ids,
        "job_embeddings": job_embeddings, "resume_embeddings": resume_embeddings,
        "queries": query_table, "candidates": candidate_table,
        "matcher": matcher, "index": index,
    }
    return rankings, context

def run_faiss_experiment(split_frame: pd.DataFrame, artifact_prefix: str = "faiss_full_test_metrics") -> dict[str, Any]:
    """
    Standard wrapper to run the FAISS experiment and log metrics.
    """
    # Assuming these helpers are defined globally in your script
    queries = derive_unique_job_queries(split_frame)
    candidates = derive_unique_resume_candidates(split_frame)
    
    start = time.perf_counter()
    
    # Initialize your SBERT Matcher (ensure this class is already defined)
    matcher = NotebookSBERTMatcher()
    
    rankings, context = rank_with_faiss(queries, candidates, matcher, model_name="faiss")
    
    runtime_seconds = time.perf_counter() - start
    print(f"FAISS Experiment Runtime: {runtime_seconds:.2f}s")
    
    # Evaluate and Save
    aggregate_metrics, query_metrics = evaluate_rankings(rankings, split_frame)
    save_metric_bundle(artifact_prefix, aggregate_metrics, query_metrics, rankings=rankings)
    
    return {
        "rankings": rankings,
        "aggregate_metrics": aggregate_metrics,
        "query_metrics": query_metrics,
        "context": context,
        "runtime_seconds": runtime_seconds,
    }
