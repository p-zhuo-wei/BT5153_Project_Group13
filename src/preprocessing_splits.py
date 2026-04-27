"""Split assignment + dataset summary writers + top-level dataset builder.

Cell-equivalent: Step 1C (cell 12).
"""

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR, SPLITS_DIR, TABLES_DIR, RANDOM_SEED,
)
from src.preprocessing_text import load_spacy_model
from src.preprocessing_pairs import (
    load_resume_data, load_job_data, build_pairs,
)


# Split assignment, saved summaries, and the top-level dataset builder.

def assign_splits(resumes: pd.DataFrame, pairs: pd.DataFrame, train_size: float, val_size: float, test_size: float, seed: int) -> pd.DataFrame:
    if not math.isclose(train_size + val_size + test_size, 1.0, rel_tol=1e-6):
        raise ValueError("train/val/test sizes must sum to 1.0")

    resume_split_frame = resumes[["resume_id", "normalized_category"]].drop_duplicates().copy()
    train_ids, temp_ids = train_test_split(
        resume_split_frame,
        test_size=(val_size + test_size),
        stratify=resume_split_frame["normalized_category"],
        random_state=seed,
    )
    relative_test = test_size / (val_size + test_size)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test,
        stratify=temp_ids["normalized_category"],
        random_state=seed,
    )

    split_lookup = {
        **{resume_id: "train" for resume_id in train_ids["resume_id"].tolist()},
        **{resume_id: "val" for resume_id in val_ids["resume_id"].tolist()},
        **{resume_id: "test" for resume_id in test_ids["resume_id"].tolist()},
    }
    pairs = pairs.copy()
    pairs["split"] = pairs["resume_id"].map(split_lookup)
    return pairs


def summarize_outputs(resumes: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, object]:
    return {
        "resumes_processed": int(len(resumes)),
        "jobs_processed": int(len(jobs)),
        "pairs_processed": int(len(pairs)),
        "resume_categories": resumes["normalized_category"].value_counts().sort_index().to_dict(),
        "job_categories": jobs["normalized_category"].value_counts().sort_index().to_dict(),
        "label_distribution": pairs["relevance_label"].value_counts().sort_index().to_dict(),
        "split_distribution": pairs["split"].value_counts().sort_index().to_dict(),
        "mean_resume_length": round(float(resumes["resume_text_length"].mean()), 2),
        "mean_job_length": round(float(jobs["job_text_length"].mean()), 2),
    }


def save_dataset_outputs(resumes: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame, summary: dict[str, object]) -> None:
    resumes_out = resumes.copy()
    jobs_out = jobs.copy()
    pairs_out = pairs.copy()

    for frame in (resumes_out, jobs_out):
        frame["skills"] = frame["skills"].map(lambda value: ";".join(value))
        frame.drop(columns=["token_set"], inplace=True)

    resumes_out.to_csv(PROCESSED_DATA_DIR / "resumes_processed.csv", index=False)
    jobs_out.to_csv(PROCESSED_DATA_DIR / "jobs_processed.csv", index=False)
    pairs_out.to_csv(PROCESSED_DATA_DIR / "resume_job_pairs.csv", index=False)
    (PROCESSED_DATA_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    pd.DataFrame(sorted(summary["resume_categories"].items()), columns=["category", "resume_count"]).to_csv(PROCESSED_DATA_DIR / "resume_category_counts.csv", index=False)
    pd.DataFrame(sorted(summary["job_categories"].items()), columns=["category", "job_count"]).to_csv(PROCESSED_DATA_DIR / "job_category_counts.csv", index=False)
    pd.DataFrame(sorted(summary["label_distribution"].items()), columns=["relevance_label", "count"]).to_csv(TABLES_DIR / "pair_label_counts.csv", index=False)
    pd.DataFrame([
        {"asset": "resumes", "rows": len(resumes)},
        {"asset": "jobs", "rows": len(jobs)},
        {"asset": "pairs", "rows": len(pairs)},
    ]).to_csv(TABLES_DIR / "dataset_asset_counts.csv", index=False)

    for split_name, split_frame in pairs_out.groupby("split"):
        split_frame.to_csv(SPLITS_DIR / f"{split_name}.csv", index=False)


def build_dataset(resume_limit_per_category: int = 140, job_limit_per_category: int = 220, positive_per_resume: int = 3, partial_per_resume: int = 2, negative_per_resume: int = 3) -> dict[str, Any]:
    resume_path = RAW_DATA_DIR / "Resume" / "Resume.csv"
    job_path = RAW_DATA_DIR / "fake_job_postings.csv"
    if not resume_path.exists() or not job_path.exists():
        missing = [str(path.relative_to(PROJECT_ROOT)) for path in [resume_path, job_path] if not path.exists()]
        raise FileNotFoundError(f"Missing raw inputs: {missing}")

    nlp = load_spacy_model(enabled=True)
    resumes = load_resume_data(resume_path, nlp, resume_limit_per_category, RANDOM_SEED)
    jobs = load_job_data(job_path, nlp, job_limit_per_category, RANDOM_SEED)

    shared_categories = sorted(set(resumes["normalized_category"]) & set(jobs["normalized_category"]))
    resumes = resumes[resumes["normalized_category"].isin(shared_categories)].reset_index(drop=True)
    jobs = jobs[jobs["normalized_category"].isin(shared_categories)].reset_index(drop=True)

    pairs = build_pairs(
        resumes=resumes,
        jobs=jobs,
        positive_per_resume=positive_per_resume,
        partial_per_resume=partial_per_resume,
        negative_per_resume=negative_per_resume,
        seed=RANDOM_SEED,
    )
    pairs = assign_splits(resumes, pairs, 0.70, 0.15, 0.15, RANDOM_SEED)
    summary = summarize_outputs(resumes, jobs, pairs)
    save_dataset_outputs(resumes, jobs, pairs, summary)
    return {"resumes": resumes, "jobs": jobs, "pairs": pairs, "summary": summary}
