"""Candidate pool sampling and weakly labelled (job, resume) pair construction.

Cell-equivalent: Step 1B (cell 10).
"""

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.preprocessing_text import (
    clean_text, normalize_text, extract_skills, mask_pii,
    normalize_resume_category, infer_job_category, token_set, overlap_score,
    load_spacy_model,
    RESUME_CATEGORY_MAP, JOB_CATEGORY_RULES, SKILL_PATTERNS, RELATED_CATEGORIES,
)


# Dataset loading helpers and pair-construction logic.

def sample_per_category(frame: pd.DataFrame, category_col: str, limit: int, seed: int) -> pd.DataFrame:
    samples = []
    for _, group in frame.groupby(category_col, sort=True):
        if len(group) <= limit:
            samples.append(group)
        else:
            samples.append(group.sample(n=limit, random_state=seed))
    return pd.concat(samples, ignore_index=True)


def load_resume_data(path: Path, nlp, limit_per_category: int, seed: int) -> pd.DataFrame:
    resumes = pd.read_csv(path, usecols=["ID", "Resume_str", "Category"])
    resumes = resumes.rename(columns={"ID": "resume_id", "Resume_str": "resume_text_raw", "Category": "resume_category_raw"})
    resumes["normalized_category"] = resumes["resume_category_raw"].map(normalize_resume_category)
    resumes = resumes[resumes["normalized_category"] != "other"].copy()
    resumes["resume_text_clean"] = resumes["resume_text_raw"].fillna("").map(clean_text)
    resumes["resume_text"] = resumes["resume_text_clean"].map(lambda value: mask_pii(value, nlp))
    resumes["resume_text_length"] = resumes["resume_text"].str.len()
    resumes = resumes[resumes["resume_text_length"] >= 200].drop_duplicates(subset=["resume_id"])
    resumes = sample_per_category(resumes, "normalized_category", limit_per_category, seed)
    resumes["skills"] = resumes["resume_text"].map(extract_skills)
    resumes["skill_count"] = resumes["skills"].map(len)
    resumes["token_set"] = resumes["resume_text"].map(token_set)
    return resumes.reset_index(drop=True)


def compose_job_text(row: pd.Series) -> str:
    parts = [
        row.get("title", ""),
        row.get("department", ""),
        row.get("description", ""),
        row.get("requirements", ""),
        row.get("benefits", ""),
        row.get("industry", ""),
        row.get("function", ""),
    ]
    return " ".join(str(part) for part in parts if pd.notna(part) and str(part).strip())


def load_job_data(path: Path, nlp, limit_per_category: int, seed: int) -> pd.DataFrame:
    jobs = pd.read_csv(path)
    jobs = jobs[(jobs["fraudulent"] == 0) & jobs["title"].notna() & jobs["description"].notna()].copy()
    jobs["job_text_raw"] = jobs.apply(compose_job_text, axis=1)
    jobs["normalized_category"] = jobs.apply(infer_job_category, axis=1)
    jobs = jobs[jobs["normalized_category"] != "other"].copy()
    jobs["job_text_clean"] = jobs["job_text_raw"].fillna("").map(clean_text)
    jobs["job_text"] = jobs["job_text_clean"].map(lambda value: mask_pii(value, nlp))
    jobs["job_text_length"] = jobs["job_text"].str.len()
    jobs = jobs[jobs["job_text_length"] >= 150]
    jobs = jobs.drop_duplicates(subset=["title", "job_text"])
    jobs = sample_per_category(jobs, "normalized_category", limit_per_category, seed)
    jobs["skills"] = jobs["job_text"].map(extract_skills)
    jobs["skill_count"] = jobs["skills"].map(len)
    jobs["token_set"] = jobs["job_text"].map(token_set)
    return jobs.reset_index(drop=True)


def prepare_candidate_index(jobs: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    index = {}
    for category, group in jobs.groupby("normalized_category"):
        index[category] = group.to_dict("records")
    return index


def choose_jobs(resume_row: pd.Series, candidate_jobs: Iterable[dict[str, object]], k: int, seen_job_ids: set[int | str]) -> list[dict[str, object]]:
    scored = []
    resume_skills = set(resume_row["skills"])
    resume_tokens = resume_row["token_set"]
    for job in candidate_jobs:
        job_id = job["job_id"]
        if job_id in seen_job_ids:
            continue
        job_skills = set(job["skills"])
        job_tokens = job["token_set"]
        score = overlap_score(resume_skills, job_skills, resume_tokens, job_tokens)
        scored.append((score, job))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    for score, job in scored:
        if len(chosen) >= k:
            break
        job["_pair_score"] = round(score, 6)
        chosen.append(job)
        seen_job_ids.add(job["job_id"])
    return chosen


def make_pair_row(resume_row: pd.Series, job: dict[str, object], label: int, rationale: str) -> dict[str, object]:
    shared_skills = sorted(set(resume_row["skills"]) & set(job["skills"]))
    return {
        "resume_id": resume_row["resume_id"],
        "job_id": job["job_id"],
        "resume_text": resume_row["resume_text"],
        "job_text": job["job_text"],
        "resume_category": resume_row["normalized_category"],
        "job_category": job["normalized_category"],
        "resume_category_raw": resume_row["resume_category_raw"],
        "job_title": job["title"],
        "job_function": job.get("function", ""),
        "job_industry": job.get("industry", ""),
        "relevance_label": label,
        "label_rationale": rationale,
        "pair_score": job.get("_pair_score", 0.0),
        "resume_skill_count": len(resume_row["skills"]),
        "job_skill_count": len(job["skills"]),
        "shared_skills": ";".join(shared_skills),
    }


def build_pairs(resumes: pd.DataFrame, jobs: pd.DataFrame, positive_per_resume: int, partial_per_resume: int, negative_per_resume: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    job_index = prepare_candidate_index(jobs)
    all_categories = sorted(job_index)
    rows = []

    for resume in resumes.itertuples(index=False):
        resume_row = pd.Series(resume._asdict())
        category = resume_row["normalized_category"]
        seen_job_ids = set()

        positive_jobs = choose_jobs(resume_row, job_index.get(category, []), positive_per_resume, seen_job_ids)
        for job in positive_jobs:
            rows.append(make_pair_row(resume_row, job, 2, "same normalized category with highest lexical overlap"))

        related_categories = RELATED_CATEGORIES.get(category, set())
        related_candidates = [job for rel_category in related_categories for job in job_index.get(rel_category, [])]
        partial_jobs = choose_jobs(resume_row, related_candidates, partial_per_resume, seen_job_ids)
        for job in partial_jobs:
            rows.append(make_pair_row(resume_row, job, 1, "adjacent category selected via taxonomy and overlap"))

        negative_pool_categories = [value for value in all_categories if value != category and value not in related_categories]
        negative_pool = [job for neg_category in negative_pool_categories for job in job_index.get(neg_category, [])]
        if negative_pool:
            sample_size = min(negative_per_resume * 5, len(negative_pool))
            sampled_indices = rng.choice(len(negative_pool), size=sample_size, replace=False)
            sampled_candidates = [negative_pool[index] for index in sampled_indices]
            negative_jobs = choose_jobs(resume_row, sampled_candidates, negative_per_resume, seen_job_ids)
        else:
            negative_jobs = []
        for job in negative_jobs:
            rows.append(make_pair_row(resume_row, job, 0, "distant category with low semantic overlap"))

    pairs = pd.DataFrame(rows)
    pairs["pair_id"] = [f"pair_{index:07d}" for index in range(1, len(pairs) + 1)]
    ordered_columns = [
        "pair_id", "resume_id", "job_id", "resume_text", "job_text", "resume_category", "job_category",
        "resume_category_raw", "job_title", "job_function", "job_industry", "relevance_label",
        "label_rationale", "pair_score", "resume_skill_count", "job_skill_count", "shared_skills",
    ]
    return pairs[ordered_columns]
