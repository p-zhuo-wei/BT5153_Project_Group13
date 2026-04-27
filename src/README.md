# `src/` — pipeline modules

Importable Python modules extracted from `archive/group14_resume_job_matching_end_to_end_260416.ipynb`.
Each module exposes the function definitions from one section of the notebook; **none of
the modules execute at import time**. Orchestration lives in `overall.ipynb` at the
repo root, which calls each `run_*_experiment(...)` with cache-aware skipping.

| Module | Notebook source | Key exports |
|---|---|---|
| `config.py` | top-of-notebook constants | `PROJECT_ROOT`, `METRICS_DIR`, `RANDOM_SEED`, `TOP_K`, `RERANK_SHORTLIST_K`, `SBERT_MODEL_NAME`, `SPLADE_MODEL_NAME`, `CROSS_ENCODER_MODEL_NAME`, `LM_STUDIO_BASE_URL`, runtime flags |
| `preprocessing_text.py` | rule tables (cell 6) + Step 1A (cell 8) | `STOPWORDS`, `RESUME_CATEGORY_MAP`, `JOB_CATEGORY_RULES`, `SKILL_PATTERNS`, `clean_text`, `mask_pii`, `extract_skills`, `normalize_resume_category`, `infer_job_category`, `normalize_text`, `truncate_text` |
| `preprocessing_pairs.py` | Step 1B (cell 10) | `load_resume_data`, `load_job_data`, `build_pairs`, `sample_per_category` |
| `preprocessing_splits.py` | Step 1C (cell 12) | `assign_splits`, `summarize_outputs`, `save_dataset_outputs`, `build_dataset` |
| `data_loader.py` | Step 2A (cell 21) | `load_split`, `derive_unique_job_queries`, `derive_unique_resume_candidates`, `build_relevance_lookup`, `attach_relevance_labels`, `build_ranking_results` |
| `stage_1_tfidf.py` | Steps 3A + 3B | `compose_job_text`, `compose_resume_text`, `rank_with_tfidf`, `run_tfidf_experiment` |
| `stage_1_sbert.py` | Steps 4A + 4B | `NotebookSBERTMatcher`, `rank_with_sbert`, `run_sbert_experiment` |
| `stage_1_splade.py` | SPLADE section | `SPLADEVectorizer`, `rank_with_splade`, `run_splade_experiment`, `SPLADE_AVAILABLE` |
| `stage_1_faiss.py` | FAISS section | `rank_with_faiss`, `run_faiss_experiment`, `FAISS_AVAILABLE` |
| `stage_2_ce_rerank.py` | Cross-encoder section | `rerank_with_cross_encoder`, `run_cross_encoder_on_first_stage`, `CROSS_ENCODER_AVAILABLE` |
| `stage_2_llm_rerank.py` | Steps 5A–5D | `lm_studio_available`, `build_llm_messages`, `parse_llm_ranking`, `call_llm_for_shortlist`, `run_llm_experiment`, `rerank_with_llm`, `run_llm_full_test_experiment`, `run_llm_subset_experiment`, `load_saved_llm_outputs`, `_read_jsonl` |

## Conventions

- All modules use absolute imports (`from src.config import ...`, `from eval.evaluator import ...`).
- All file system paths come from `src.config`, never hard-coded.
- Each `run_*_experiment` returns a dict with the keys `aggregate_metrics`, `query_metrics`,
  `rankings`, `context`, `runtime_seconds`. The LLM additionally returns `reasons`,
  `traces`, `subset_job_ids`, `detected_model_id`. Cache-loaded variants set `rankings`
  and `context` to `None` (downstream consumers in `overall.ipynb` only read metrics).
