"""Perturbation rules + ranking-stability fairness audit.

Cell-equivalents: Step 7A (cell 65) + 7B (cell 67).
"""

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import RANDOM_SEED, TOP_K
from src.data_loader import (
    derive_unique_job_queries, derive_unique_resume_candidates,
)
from src.stage_1_tfidf import rank_with_tfidf
from src.stage_1_sbert import NotebookSBERTMatcher, rank_with_sbert


# Perturbation dictionaries and text-edit helpers for the fairness audit.

GENDER_TO_FEMININE = {
    "he": "she", "him": "her", "his": "her", "himself": "herself", "man": "woman", "men": "women",
    "male": "female", "father": "mother", "brother": "sister", "son": "daughter", "mr": "ms",
    "salesman": "saleswoman", "chairman": "chairwoman",
}
GENDER_TO_MASCULINE = {
    "she": "he", "her": "his", "hers": "his", "herself": "himself", "woman": "man", "women": "men",
    "female": "male", "mother": "father", "sister": "brother", "daughter": "son", "ms": "mr",
    "saleswoman": "salesman", "chairwoman": "chairman",
}
NAME_GROUPS = {
    "white_american": [("Emily", "Smith"), ("Brad", "Miller"), ("Katie", "Johnson"), ("Greg", "Wilson")],
    "african_american": [("Aisha", "Jackson"), ("Darnell", "Washington"), ("Imani", "Jefferson"), ("Malik", "Robinson")],
    "south_asian": [("Priya", "Patel"), ("Arjun", "Sharma"), ("Neha", "Reddy"), ("Rahul", "Menon")],
    "east_asian": [("Mei", "Chen"), ("Jun", "Park"), ("Yuna", "Kim"), ("Wei", "Lin")],
}
MONTH_PATTERN = r"(?:jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)"


def preserve_case(source: str, replacement: str) -> str:
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement.capitalize()
    return replacement


def replace_terms(text: str, replacements: dict[str, str]) -> str:
    if not text:
        return text
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, replacements), key=len, reverse=True)) + r")\b", flags=re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        original = match.group(0)
        replacement = replacements.get(original.lower(), original)
        return preserve_case(original, replacement)

    return pattern.sub(repl, text)


def deterministic_name(row: pd.Series, group_key: str) -> str:
    pool = NAME_GROUPS[group_key]
    stable_value = f"{group_key}|{row.get('resume_id', '')}|{row.get('resume_category', '')}"
    index = abs(hash(stable_value)) % len(pool)
    first_name, last_name = pool[index]
    return f"{first_name} {last_name}"


def swap_or_insert_name(text: str, replacement_name: str) -> str:
    if not text:
        return f"Candidate name: {replacement_name}."
    patterns = [
        re.compile(r"^(?:name\s*[:\-]\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b"),
        re.compile(r"^([A-Z][a-z]+\s+[A-Z][a-z]+)(?=\s+(?:summary|profile|objective|experience|curriculum|resume)\b)", flags=re.IGNORECASE),
    ]
    for pattern in patterns:
        if pattern.search(text):
            return pattern.sub(lambda match: match.group(0).replace(match.group(1), replacement_name, 1), text, count=1)
    return f"Candidate name: {replacement_name}. {text}"


def apply_age_reduction(text: str) -> str:
    updated = text
    substitutions = [
        (re.compile(r"\b(?:age|aged)\s*[:\-]?\s*\d{1,2}\b", flags=re.IGNORECASE), ""),
        (re.compile(r"\b(?:date\s+of\s+birth|dob|birth\s*date)\s*[:\-]?\s*[^\n\r.;]+", flags=re.IGNORECASE), ""),
        (re.compile(rf"\b{MONTH_PATTERN}\s+\d{{4}}\s*(?:to|-|–)\s*(?:present|current|{MONTH_PATTERN}\s+\d{{4}}|\d{{4}})\b", flags=re.IGNORECASE), "employment period"),
        (re.compile(r"\b(?:19|20)\d{2}\b"), ""),
        (re.compile(r"\b\d{1,2}\+?\s+years?\b", flags=re.IGNORECASE), "years"),
        (re.compile(r"\bmore than\s+\d{1,2}\s+years?\b", flags=re.IGNORECASE), "substantial experience"),
    ]
    for pattern, replacement in substitutions:
        updated = pattern.sub(replacement, updated)
    return re.sub(r"\s+", " ", updated).strip()


def build_perturbation_specs() -> dict[str, Any]:
    return {
        "gender_feminine": lambda text, row: replace_terms(text, GENDER_TO_FEMININE),
        "gender_masculine": lambda text, row: replace_terms(text, GENDER_TO_MASCULINE),
        "name_white_american": lambda text, row: swap_or_insert_name(text, deterministic_name(row, "white_american")),
        "name_african_american": lambda text, row: swap_or_insert_name(text, deterministic_name(row, "african_american")),
        "name_south_asian": lambda text, row: swap_or_insert_name(text, deterministic_name(row, "south_asian")),
        "name_east_asian": lambda text, row: swap_or_insert_name(text, deterministic_name(row, "east_asian")),
        "age_reduced": lambda text, row: apply_age_reduction(text),
    }

# Fairness audit runner and ranking-stability comparison logic.

def subset_split(frame: pd.DataFrame, max_jobs: int = 25, max_resumes: int = 50) -> pd.DataFrame:
    job_ids = frame["job_id"].drop_duplicates().sample(n=min(max_jobs, frame["job_id"].nunique()), random_state=RANDOM_SEED).tolist()
    sampled = frame.loc[frame["job_id"].isin(job_ids)].copy()
    resume_ids = sampled["resume_id"].drop_duplicates().sample(n=min(max_resumes, sampled["resume_id"].nunique()), random_state=RANDOM_SEED + 1).tolist()
    return sampled.loc[sampled["resume_id"].isin(resume_ids)].reset_index(drop=True)


def compare_rankings(base_rankings: pd.DataFrame, perturbed_rankings: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    rows = []
    for job_id in sorted(base_rankings["job_id"].unique().tolist()):
        base_group = base_rankings.loc[base_rankings["job_id"] == job_id].sort_values("rank")
        perturbed_group = perturbed_rankings.loc[perturbed_rankings["job_id"] == job_id].sort_values("rank")
        merged = base_group[["resume_id", "rank"]].merge(
            perturbed_group[["resume_id", "rank"]],
            on="resume_id",
            suffixes=("_base", "_perturbed"),
        )
        top_base = base_group.head(top_k)["resume_id"].tolist()
        top_perturbed = perturbed_group.head(top_k)["resume_id"].tolist()
        top1_change = int(top_base[:1] != top_perturbed[:1])
        topk_overlap = len(set(top_base) & set(top_perturbed)) / max(top_k, 1)
        rho = spearmanr(merged["rank_base"], merged["rank_perturbed"]).statistic if len(merged) > 1 else 1.0
        mean_displacement = float((merged["rank_base"] - merged["rank_perturbed"]).abs().mean()) if len(merged) else 0.0
        rows.append(
            {
                "job_id": job_id,
                "top1_change": top1_change,
                "top10_overlap": topk_overlap,
                "spearman": float(0.0 if pd.isna(rho) else rho),
                "mean_rank_displacement": mean_displacement,
            }
        )
    return pd.DataFrame(rows)


def run_fairness_audit(split_frame: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled = subset_split(split_frame)
    queries = derive_unique_job_queries(sampled).sort_values("job_id").reset_index(drop=True)
    candidates = derive_unique_resume_candidates(sampled).sort_values("resume_id").reset_index(drop=True)
    perturbations = build_perturbation_specs()

    fairness_matcher = None
    if model_name == "tfidf":
        base_rankings, _ = rank_with_tfidf(queries, candidates, model_name="tfidf_fairness")
    elif model_name == "sbert":
        fairness_matcher = NotebookSBERTMatcher()  # load once, reuse for all perturbations
        base_rankings, _ = rank_with_sbert(queries, candidates, fairness_matcher, model_name="sbert_fairness")
    else:
        raise ValueError("Supported fairness models are 'tfidf' and 'sbert'.")

    job_rows = []
    summary_rows = []
    for perturbation_name, perturbation_fn in perturbations.items():
        perturbed_candidates = candidates.copy()
        perturbed_candidates["resume_text"] = [perturbation_fn(text, row) for text, (_, row) in zip(perturbed_candidates["resume_text"].tolist(), perturbed_candidates.iterrows(), strict=False)]
        if model_name == "tfidf":
            perturbed_rankings, _ = rank_with_tfidf(queries, perturbed_candidates, model_name="tfidf_fairness")
        else:
            perturbed_rankings, _ = rank_with_sbert(queries, perturbed_candidates, fairness_matcher, model_name="sbert_fairness")
        comparison = compare_rankings(base_rankings, perturbed_rankings)
        comparison.insert(0, "perturbation", perturbation_name)
        comparison.insert(0, "model", model_name.upper())
        job_rows.append(comparison)
        summary_rows.append(
            {
                "model": model_name.upper(),
                "perturbation": perturbation_name,
                "top1_change_rate": float(comparison["top1_change"].mean()),
                "top10_overlap": float(comparison["top10_overlap"].mean()),
                "mean_spearman": float(comparison["spearman"].mean()),
                "mean_rank_displacement": float(comparison["mean_rank_displacement"].mean()),
            }
        )
    return pd.DataFrame(summary_rows), pd.concat(job_rows, ignore_index=True)
