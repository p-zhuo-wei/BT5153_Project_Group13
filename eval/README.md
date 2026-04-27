# `eval/` — evaluation and fairness modules

Pure functions, no execution at import time. Called from `overall.ipynb`.

| Module | Notebook source | Key exports |
|---|---|---|
| `metrics.py` | Step 2B (cell 23) | `precision_at_k`, `recall_at_k`, `reciprocal_rank`, `average_precision`, `dcg_at_k`, `ndcg_at_k` |
| `evaluator.py` | Step 2C (cell 25) | `evaluate_rankings`, `save_metric_bundle`, `paired_bootstrap_significance_test` |
| `fairness.py` | Steps 7A + 7B | `GENDER_TO_FEMININE`, `GENDER_TO_MASCULINE`, `build_perturbation_specs`, `subset_split`, `compare_rankings`, `run_fairness_audit` |

## Notes

- `evaluator.py` writes to `METRICS_DIR` via `save_metric_bundle` — the same target every
  `run_*_experiment` in `src/` writes to. The orchestrator's cache check in `overall.ipynb`
  reads back from this same directory.
- `fairness.py`'s `run_fairness_audit` currently supports `model_name in ("tfidf", "sbert")`.
  LLM fairness was intentionally left out per the report's caveat (too thinly evaluated).

