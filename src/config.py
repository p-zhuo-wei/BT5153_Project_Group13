"""Project-wide configuration: paths, seeds, model names, runtime flags.

All other modules import from this file. The orchestrator notebook should
`from src.config import *` (or import individual names) at the top.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns


warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 140)
pd.set_option("display.max_columns", 40)
sns.set_theme(style="whitegrid", context="talk")


def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root from the current working directory.")


PROJECT_ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
TABLES_DIR = RESULTS_DIR / "tables"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"

for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR, TABLES_DIR, EMBEDDINGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TOP_K = 10
RERANK_SHORTLIST_K = 50  # top-K from first-stage retriever to feed into reranker
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPLADE_MODEL_NAME = "naver/splade-v3"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL_ID = "google/gemma-4-e2b"

RUN_LLM_SUBSET = os.getenv("RUN_LLM_SUBSET", "true").strip().lower() == "true"
RUN_LLM_FULL_TEST = os.getenv("RUN_LLM_FULL_TEST", "false").strip().lower() == "true"
RUN_FAIRNESS = os.getenv("RUN_FAIRNESS", "true").strip().lower() == "true"
RUN_LLM_FAIRNESS_SMOKE = os.getenv("RUN_LLM_FAIRNESS_SMOKE", "false").strip().lower() == "true"
LLM_SUBSET_QUERY_COUNT = int(os.getenv("LLM_SUBSET_QUERY_COUNT", "100"))
LLM_SHORTLIST_SIZE = int(os.getenv("LLM_SHORTLIST_SIZE", "20"))
LLM_JOB_TEXT_CHARS = int(os.getenv("LLM_JOB_TEXT_CHARS", "650"))
LLM_RESUME_TEXT_CHARS = int(os.getenv("LLM_RESUME_TEXT_CHARS", "550"))
LLM_MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "500"))
LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "300"))
LLM_INCLUDE_REASONS = os.getenv("LLM_INCLUDE_REASONS", "false").strip().lower() == "true"
LLM_CHECKPOINT_EVERY = int(os.getenv("LLM_CHECKPOINT_EVERY", "10"))
_llm_full_limit_raw = os.getenv("LLM_FULL_TEST_QUERY_LIMIT", "").strip()
LLM_FULL_TEST_QUERY_LIMIT = int(_llm_full_limit_raw) if _llm_full_limit_raw else None

np.random.seed(RANDOM_SEED)
