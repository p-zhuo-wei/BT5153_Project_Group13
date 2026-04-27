"""Split file loading + relevance lookups + query/candidate derivation.

Cell-equivalent: Step 2A (cell 21).
"""

from typing import Any

import numpy as np
import pandas as pd

from src.config import SPLITS_DIR


# Data-loading and lookup helpers shared by all model families.

def load_split(split_name: str) -> pd.DataFrame:
    split_path = SPLITS_DIR / f"{split_name}.csv"
    frame = pd.read_csv(split_path)
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["relevance_label"] = pd.to_numeric(frame["relevance_label"], errors="raise").astype(int)
    return frame


def derive_unique_job_queries(split_frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["job_id", "job_text", "job_title", "job_function", "job_industry", "job_category", "split"]
    return split_frame[[column for column in columns if column in split_frame.columns]].drop_duplicates(subset=["job_id"]).reset_index(drop=True)


def derive_unique_resume_candidates(split_frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["resume_id", "resume_text", "resume_category", "resume_category_raw", "resume_skill_count", "split"]
    return split_frame[[column for column in columns if column in split_frame.columns]].drop_duplicates(subset=["resume_id"]).reset_index(drop=True)


def build_relevance_lookup(split_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[object, dict[object, int]]]:
    deduped = (
        split_frame[["job_id", "resume_id", "relevance_label"]]
        .dropna(subset=["job_id", "resume_id"])
        .groupby(["job_id", "resume_id"], as_index=False)["relevance_label"]
        .max()
    )
    matrix = deduped.pivot(index="job_id", columns="resume_id", values="relevance_label").fillna(0).astype(int)
    lookup = {
        job_id: {resume_id: int(label) for resume_id, label in row.items() if int(label) > 0}
        for job_id, row in matrix.iterrows()
    }
    return matrix, lookup


def build_ranking_results(scored_pairs: pd.DataFrame, model_name: str, split: str) -> pd.DataFrame:
    rankings = scored_pairs[["job_id", "resume_id", "score"]].copy()
    rankings.insert(0, "split", split)
    rankings.insert(0, "model_name", model_name)
    rankings = rankings.sort_values(["model_name", "split", "job_id", "score", "resume_id"], ascending=[True, True, True, False, True])
    rankings["rank"] = rankings.groupby(["model_name", "split", "job_id"]).cumcount() + 1
    return rankings[["model_name", "split", "job_id", "rank", "resume_id", "score"]].reset_index(drop=True)


def attach_relevance_labels(rankings: pd.DataFrame, split_frame: pd.DataFrame) -> pd.DataFrame:
    if "relevance_label" in rankings.columns:
        labeled = rankings.copy()
        labeled['relevance_label'] = labeled['relevance_label'].fillna(0).astype(int)
        return labeled
    labels = split_frame[["job_id", "resume_id", "relevance_label"]].groupby(["job_id", "resume_id"], as_index=False)["relevance_label"].max()
    labeled = rankings.merge(labels, on=["job_id", "resume_id"], how="left")
    labeled["relevance_label"] = labeled["relevance_label"].fillna(0).astype(int)
    return labeled
