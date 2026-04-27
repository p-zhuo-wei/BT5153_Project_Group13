"""LLM second-stage reranker via LM Studio: parsing, prompts, runner, fallback.

Cell-equivalents: Step 5A (48) + 5B (50) + 5C (52, execution trimmed) + 5D (54, execution trimmed).
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

from src.config import (
    LM_STUDIO_BASE_URL, LM_STUDIO_MODEL_ID, METRICS_DIR, TABLES_DIR, TOP_K,
    LLM_CHECKPOINT_EVERY, LLM_FULL_TEST_QUERY_LIMIT, LLM_INCLUDE_REASONS,
    LLM_JOB_TEXT_CHARS, LLM_MAX_OUTPUT_TOKENS, LLM_REQUEST_TIMEOUT,
    LLM_RESUME_TEXT_CHARS, LLM_SHORTLIST_SIZE, LLM_SUBSET_QUERY_COUNT,
    PROJECT_ROOT, RANDOM_SEED,
)
from src.data_loader import (
    derive_unique_job_queries, derive_unique_resume_candidates,
)
from src.preprocessing_text import normalize_text, truncate_text
from src.stage_1_tfidf import compose_job_text, compose_resume_text
from src.data_loader import build_ranking_results
from eval.evaluator import evaluate_rankings, save_metric_bundle


# LM Studio availability checks, model detection, and structured-output parsing helpers.

def lm_studio_available(base_url: str = LM_STUDIO_BASE_URL) -> bool:
    """Check whether LM Studio is reachable at the configured base URL."""
    try:
        import requests
        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
        return response.ok
    except Exception:
        return False


def detect_lm_studio_model(base_url: str = LM_STUDIO_BASE_URL) -> str | None:
    """Query the LM Studio /v1/models endpoint and return the loaded model identifier.
    
    Returns the model ID string if LM Studio is reachable and has a model loaded,
    or None if the endpoint is unavailable. This allows the notebook to label
    LLM outputs with the actual model that produced them rather than a hardcoded name.
    """
    try:
        import requests
        response = requests.get(f"{base_url.rstrip('/')}/models", timeout=10)
        if not response.ok:
            return None
        payload = response.json()
        models = payload.get("data", [])
        if models and isinstance(models, list):
            # Return the first loaded model's ID
            return models[0].get("id", None)
        return None
    except Exception:
        return None


# Detect the actual model running in LM Studio at notebook startup.
# Falls back to the configured default if detection fails.
_detected_model = detect_lm_studio_model()
DETECTED_LLM_MODEL_ID = _detected_model if _detected_model else LM_STUDIO_MODEL_ID
if _detected_model and _detected_model != LM_STUDIO_MODEL_ID:
    print(f"Note: LM Studio is serving '{_detected_model}' (detected automatically). "
          f"The configured default was '{LM_STUDIO_MODEL_ID}'. Using the detected model.")
elif _detected_model:
    print(f"LM Studio model detected: {_detected_model}")


def extract_json_candidate(text: str) -> str | None:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
    return None


def coerce_ranked_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("ranked_candidates", "ranking", "candidates", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def parse_llm_ranking(response_text: str, lexical_ids: list[Any]) -> tuple[list[dict[str, Any]], str]:
    candidate_lookup = {str(candidate_id): candidate_id for candidate_id in lexical_ids}

    def finalize(items: Iterable[dict[str, Any]], strategy: str) -> tuple[list[dict[str, Any]], str]:
        seen = set()
        normalized = []
        for item in items:
            raw_id = item.get("resume_id") or item.get("candidate_id") or item.get("id")
            if raw_id is None:
                continue
            canonical_id = candidate_lookup.get(str(raw_id))
            if canonical_id is None or canonical_id in seen:
                continue
            seen.add(canonical_id)
            normalized.append(
                {
                    "resume_id": canonical_id,
                    "score": float(item.get("score", len(lexical_ids) - len(normalized))),
                    "reason": str(item.get("reason") or item.get("rationale") or "").strip(),
                }
            )
        if len(normalized) < len(lexical_ids):
            for candidate_id in lexical_ids:
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                normalized.append(
                    {
                        "resume_id": candidate_id,
                        "score": float(len(lexical_ids) - len(normalized)),
                        "reason": "fallback_lexical_order",
                    }
                )
        return normalized, strategy

    try:
        payload = json.loads(response_text)
        items = coerce_ranked_items(payload)
        if items:
            return finalize(items, "direct_json")
    except json.JSONDecodeError:
        pass

    json_candidate = extract_json_candidate(response_text)
    if json_candidate:
        try:
            payload = json.loads(json_candidate)
            items = coerce_ranked_items(payload)
            if items:
                return finalize(items, "embedded_json")
        except json.JSONDecodeError:
            pass

    regex_items = []
    seen_regex = set()
    for match in re.finditer(r'"(?:resume_id|candidate_id|id)"\s*:\s*"?([0-9]+)"?', response_text):
        canonical_id = candidate_lookup.get(match.group(1))
        if canonical_id is None or canonical_id in seen_regex:
            continue
        seen_regex.add(canonical_id)
        regex_items.append({"resume_id": canonical_id, "score": float(len(lexical_ids) - len(regex_items))})
    if regex_items:
        return finalize(regex_items, "regex_ids")

    lexical_items = [{"resume_id": candidate_id, "score": float(len(lexical_ids) - index), "reason": "lexical_fallback"} for index, candidate_id in enumerate(lexical_ids)]
    return finalize(lexical_items, "lexical_fallback")

# Prompt assembly and lexical shortlist generation for LLM reranking.

def build_llm_messages(query_row: pd.Series, shortlist_rows: pd.DataFrame, job_text_chars: int, resume_text_chars: int) -> list[dict[str, Any]]:
    job_summary = {
        "job_id": query_row["job_id"],
        "job_title": normalize_text(query_row.get("job_title", "")),
        "job_category": normalize_text(query_row.get("job_category", "")),
        "job_function": normalize_text(query_row.get("job_function", "")),
        "job_industry": normalize_text(query_row.get("job_industry", "")),
        "job_text": truncate_text(query_row.get("job_text", ""), job_text_chars),
    }
    candidates = []
    for _, row in shortlist_rows.iterrows():
        candidates.append(
            {
                "resume_id": row["resume_id"],
                "resume_category": normalize_text(row.get("resume_category", "")),
                "lexical_score": round(float(row.get("lexical_score", 0.0)), 6),
                "resume_text": truncate_text(row.get("resume_text", ""), resume_text_chars),
            }
        )

    system_prompt = (
        "You rerank resume candidates for a single job posting. "
        "Output the JSON answer directly. Do not show any reasoning or chain-of-thought. "
        "Return valid JSON only. Do not add markdown or extra prose."
    )
    user_prompt = {
        "task": "Rank the shortlisted resumes from best to worst job fit.",
        "scoring_guidance": [
            "Prioritise skill match, relevant experience, domain alignment, and seniority fit.",
            "Penalise missing competencies, wrong specialisation, or weak evidence.",
            "Use only the provided candidates and preserve their resume_id values exactly.",
        ],
        "output_schema": {
            "ranked_candidates": [
                ({
                    "resume_id": "string or integer copied from the shortlist",
                    "score": "number from 0 to 100; higher means better fit",
                    "reason": "short evidence-based rationale under 8 words",
                } if LLM_INCLUDE_REASONS else {
                    "resume_id": "string or integer copied from the shortlist",
                    "score": "number from 0 to 100; higher means better fit",
                })
            ]
        },
        "constraints": {
            "return_all_candidates": True,
            "ranked_candidates_length_must_equal": len(candidates),
            "unique_resume_ids_only": True,
            "sort_descending_by_fit": True,
        },
        "job": job_summary,
        "candidates": candidates,
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=True)},
    ]


def build_lexical_shortlist(queries: pd.DataFrame, candidates: pd.DataFrame, shortlist_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_table = queries.copy().reset_index(drop=True)
    candidate_table = candidates.copy().reset_index(drop=True)
    query_text = compose_job_text(query_table)
    candidate_text = compose_resume_text(candidate_table)
    vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=40000, sublinear_tf=True)
    corpus = pd.concat([query_text, candidate_text], axis=0, ignore_index=True)
    vectorizer.fit(corpus.tolist())
    query_matrix = vectorizer.transform(query_text.tolist())
    candidate_matrix = vectorizer.transform(candidate_text.tolist())
    score_matrix = (query_matrix @ candidate_matrix.T).toarray()

    shortlist_rows = []
    lexical_rows = []
    for query_index, job_id in enumerate(query_table["job_id"].tolist()):
        scores = score_matrix[query_index]
        ranking_order = np.argsort(-scores, kind="mergesort")
        lexical_rows.append(pd.DataFrame({
            "job_id": job_id,
            "resume_id": candidate_table["resume_id"].to_numpy()[ranking_order],
            "lexical_score": scores[ranking_order],
        }))
        top_indices = ranking_order[:shortlist_size]
        shortlist = candidate_table.iloc[top_indices].copy()
        shortlist.insert(0, "job_id", job_id)
        shortlist["lexical_score"] = scores[top_indices]
        shortlist_rows.append(shortlist)

    return pd.concat(shortlist_rows, ignore_index=True), pd.concat(lexical_rows, ignore_index=True)


def call_llm_for_shortlist(messages: list[dict[str, Any]], temperature: float = 0.0, max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS, request_timeout: float = LLM_REQUEST_TIMEOUT) -> str:
    """Send a reranking request to LM Studio using the auto-detected model ID."""
    import requests

    # Use the dynamically detected model ID rather than the hardcoded default.
    # `reasoning` + `thinking` silence chain-of-thought tokens on reasoning-native
    # models (e.g. gemma-4-e2b) so the entire token budget is spent on the JSON
    # answer; harmless on non-reasoning models that simply ignore them.
    response = requests.post(
        f"{LM_STUDIO_BASE_URL.rstrip('/')}/chat/completions",
        json={
            "model": DETECTED_LLM_MODEL_ID,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "reasoning": {"effort": "minimal"},
            "thinking": False,
        },
        timeout=request_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

def build_shortlist_from_rankings(
    first_stage_rankings: pd.DataFrame,
    candidates: pd.DataFrame,
    shortlist_size: int = LLM_SHORTLIST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an LLM-compatible shortlist from any first-stage rankings.

    Returns:
        shortlist_rows: DataFrame with shortlisted candidates per job (for LLM prompt).
        first_stage_order: DataFrame with full first-stage ordering (for fallback).
    """
    candidate_table = candidates.copy().reset_index(drop=True)
    candidate_table["candidate_text"] = compose_resume_text(candidate_table)
    resume_text_lookup = dict(zip(candidate_table["resume_id"].to_numpy(), candidate_table["candidate_text"].fillna("").astype(str)))
    resume_cat_lookup = dict(zip(candidate_table["resume_id"].to_numpy(), candidate_table.get("resume_category", pd.Series("", index=candidate_table.index)).fillna("").astype(str)))

    shortlist_rows = []
    order_rows = []
    for job_id in first_stage_rankings["job_id"].unique():
        job_rankings = first_stage_rankings.loc[
            first_stage_rankings["job_id"] == job_id
        ].sort_values("rank")

        for _, row in job_rankings.iterrows():
            order_rows.append({
                "job_id": job_id,
                "resume_id": row["resume_id"],
                "lexical_score": row["score"],
            })

        top_n = job_rankings.head(shortlist_size)
        for _, row in top_n.iterrows():
            rid = row["resume_id"]
            shortlist_rows.append({
                "job_id": job_id,
                "resume_id": rid,
                "resume_category": resume_cat_lookup.get(rid, ""),
                "lexical_score": row["score"],
                "resume_text": resume_text_lookup.get(rid, ""),
            })

    return pd.DataFrame(shortlist_rows), pd.DataFrame(order_rows)


def rerank_with_llm(
    first_stage_outputs: dict[str, Any],
    split_frame: pd.DataFrame,
    first_stage_label: str,
    *,
    max_queries: int | None = None,
    artifact_prefix: str | None = None,
    reasons_filename: str = "llm_rerank_reasons.csv",
    model_name: str | None = None,
) -> dict[str, Any] | None:
    """Run LLM reranking on any first-stage model's shortlist.

    Uses the same LM Studio infrastructure as the original LLM section,
    but pulls the shortlist from the provided first-stage rankings instead of
    always using TF-IDF.
    """
    if not lm_studio_available():
        print(f"LM Studio not reachable. Skipping LLM rerank on {first_stage_label}.")
        return None

    if model_name is None:
        model_name = f"llm_rerank_{first_stage_label}"

    queries = derive_unique_job_queries(split_frame).sort_values("job_id").reset_index(drop=True)
    candidates = derive_unique_resume_candidates(split_frame).sort_values("resume_id").reset_index(drop=True)
    if max_queries is not None:
        queries = queries.head(max_queries).copy()

    # Filter first-stage rankings to only the selected queries
    first_stage_rankings = first_stage_outputs["rankings"]
    if max_queries is not None:
        selected_job_ids = set(queries["job_id"].tolist())
        first_stage_rankings = first_stage_rankings.loc[first_stage_rankings["job_id"].isin(selected_job_ids)]

    shortlist_rows, first_stage_order = build_shortlist_from_rankings(
        first_stage_rankings, candidates,
    )

    artifact_stem = (artifact_prefix or model_name).replace("_metrics", "")
    raw_rankings_path = METRICS_DIR / f"{artifact_stem}_checkpoint_rankings_raw.csv"
    trace_path = METRICS_DIR / f"{artifact_stem}_traces.jsonl"
    reasons_path = TABLES_DIR / reasons_filename

    _remove_if_exists(raw_rankings_path)
    _remove_if_exists(trace_path)
    _remove_if_exists(reasons_path)

    start = time.perf_counter()
    target_jobs = queries["job_id"].tolist()
    processed_job_ids: set[int] = set()

    progress = tqdm(
        list(queries.iterrows()),
        total=len(queries),
        desc=f"LLM rerank ({first_stage_label})",
        unit="query",
    )
    latency_running_total = 0.0
    for index, (_, query_row) in enumerate(progress, start=1):
        job_id = int(query_row["job_id"])
        if job_id in processed_job_ids:
            progress.update(0)
            continue

        query_shortlist = shortlist_rows.loc[shortlist_rows["job_id"] == query_row["job_id"]].copy()
        fs_order = first_stage_order.loc[first_stage_order["job_id"] == query_row["job_id"]].copy()
        fs_ids = query_shortlist["resume_id"].tolist()
        messages = build_llm_messages(query_row, query_shortlist, LLM_JOB_TEXT_CHARS, LLM_RESUME_TEXT_CHARS)
        query_start = time.perf_counter()
        response_text = call_llm_for_shortlist(messages)
        latency_seconds = time.perf_counter() - query_start
        latency_running_total += latency_seconds
        progress.set_postfix(avg_s_per_query=f"{latency_running_total / max(index, 1):.1f}")
        parsed_items, strategy = parse_llm_ranking(response_text, fs_ids)
        ordered_shortlist = [item["resume_id"] for item in parsed_items]

        remainder = [rid for rid in fs_order["resume_id"].tolist() if rid not in ordered_shortlist]
        ordered_all = ordered_shortlist + remainder
        score_lookup = {rid: float(len(ordered_all) - rank_idx) for rank_idx, rid in enumerate(ordered_all)}
        query_rank_rows = pd.DataFrame({
            "job_id": job_id,
            "resume_id": ordered_all,
            "score": [score_lookup[rid] for rid in ordered_all],
        })
        query_rank_rows.to_csv(raw_rankings_path, mode="a", header=not raw_rankings_path.exists(), index=False)

        reason_rows = pd.DataFrame([{
            "job_id": job_id,
            "resume_id": item["resume_id"],
            "shortlist_rank": pos,
            "reason": item.get("reason", ""),
            "parse_strategy": strategy,
        } for pos, item in enumerate(parsed_items, start=1)])
        if not reason_rows.empty:
            reason_rows.to_csv(reasons_path, mode="a", header=not reasons_path.exists(), index=False)

        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "job_id": job_id,
                "first_stage": first_stage_label,
                "parse_strategy": strategy,
                "candidate_ids": fs_ids,
                "ordered_ids": ordered_shortlist,
                "latency_seconds": latency_seconds,
                "response_text": response_text,
            }, ensure_ascii=True) + "\n")

        processed_job_ids.add(job_id)
        if LLM_CHECKPOINT_EVERY > 0 and (len(processed_job_ids) % LLM_CHECKPOINT_EVERY == 0 or len(processed_job_ids) == len(target_jobs)):
            print(f"[LLM+{first_stage_label}] Processed {len(processed_job_ids)}/{len(target_jobs)} queries")

    runtime_seconds = time.perf_counter() - start
    if not raw_rankings_path.exists():
        return None

    scored_pairs = pd.read_csv(raw_rankings_path)
    rankings = build_ranking_results(scored_pairs, model_name=model_name, split="test")
    eval_split = split_frame.loc[split_frame["job_id"].isin(target_jobs)].copy()
    aggregate_metrics, query_metrics = evaluate_rankings(rankings, eval_split)
    if artifact_prefix:
        save_metric_bundle(artifact_prefix, aggregate_metrics, query_metrics)

    reasons = pd.read_csv(reasons_path) if reasons_path.exists() else pd.DataFrame(columns=["job_id", "resume_id", "shortlist_rank", "reason", "parse_strategy"])
    traces = pd.DataFrame(_read_jsonl(trace_path))
    if not traces.empty and "latency_seconds" in traces.columns:
        runtime_seconds = float(traces["latency_seconds"].sum())

    return {
        "rankings": rankings,
        "aggregate_metrics": aggregate_metrics,
        "query_metrics": query_metrics,
        "reasons": reasons,
        "traces": traces,
        "runtime_seconds": runtime_seconds,
        "subset_job_ids": target_jobs,
        "detected_model_id": DETECTED_LLM_MODEL_ID,
    }

# Live LLM reranking and artifact export.

def _artifact_stem(artifact_prefix: str) -> str:
    return artifact_prefix.replace("_metrics", "")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


# Build a model label from the detected LM Studio model ID for use in outputs.
# Normalises to a safe string like "llm_google_gemma_4_e2b" or "llm_meta_llama_3_8b".
def _llm_model_label() -> str:
    raw = DETECTED_LLM_MODEL_ID or "unknown"
    return "llm_" + re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def run_llm_experiment(
    split_frame: pd.DataFrame,
    *,
    max_queries: int | None,
    artifact_prefix: str,
    reasons_filename: str,
    model_name: str | None = None,
    resume: bool = True,
    checkpoint_every: int = LLM_CHECKPOINT_EVERY,
) -> dict[str, Any]:
    if not lm_studio_available():
        raise RuntimeError("LM Studio is not reachable. Start LM Studio before executing the LLM section.")

    # Use the detected model label unless overridden
    if model_name is None:
        model_name = _llm_model_label()

    queries = derive_unique_job_queries(split_frame).sort_values("job_id").reset_index(drop=True)
    candidates = derive_unique_resume_candidates(split_frame).sort_values("resume_id").reset_index(drop=True)
    if max_queries is not None:
        queries = queries.head(max_queries).copy()

    artifact_stem = _artifact_stem(artifact_prefix)
    raw_rankings_path = METRICS_DIR / f"{artifact_stem}_checkpoint_rankings_raw.csv"
    trace_path = METRICS_DIR / f"{artifact_stem}_traces.jsonl"
    reasons_path = TABLES_DIR / reasons_filename

    if not resume:
        _remove_if_exists(raw_rankings_path)
        _remove_if_exists(trace_path)
        _remove_if_exists(reasons_path)

    processed_job_ids: set[int] = set()
    if raw_rankings_path.exists():
        processed_job_ids = set(pd.read_csv(raw_rankings_path, usecols=["job_id"])["job_id"].drop_duplicates().tolist())

    shortlist_rows, lexical_rankings = build_lexical_shortlist(queries, candidates, shortlist_size=LLM_SHORTLIST_SIZE)
    start = time.perf_counter()
    target_jobs = queries["job_id"].tolist()

    progress = tqdm(
        list(queries.iterrows()),
        total=len(queries),
        desc=f"LLM rerank ({model_name})",
        unit="query",
    )
    latency_running_total = 0.0
    for index, (_, query_row) in enumerate(progress, start=1):
        job_id = int(query_row["job_id"])
        if job_id in processed_job_ids:
            progress.update(0)
            continue

        query_shortlist = shortlist_rows.loc[shortlist_rows["job_id"] == query_row["job_id"]].copy()
        lexical_full = lexical_rankings.loc[lexical_rankings["job_id"] == query_row["job_id"]].copy()
        lexical_ids = query_shortlist["resume_id"].tolist()
        messages = build_llm_messages(query_row, query_shortlist, LLM_JOB_TEXT_CHARS, LLM_RESUME_TEXT_CHARS)
        query_start = time.perf_counter()
        response_text = call_llm_for_shortlist(messages)
        latency_seconds = time.perf_counter() - query_start
        latency_running_total += latency_seconds
        progress.set_postfix(avg_s_per_query=f"{latency_running_total / max(index, 1):.1f}")
        parsed_items, strategy = parse_llm_ranking(response_text, lexical_ids)
        ordered_shortlist = [item["resume_id"] for item in parsed_items]

        remainder = [resume_id for resume_id in lexical_full["resume_id"].tolist() if resume_id not in ordered_shortlist]
        ordered_all = ordered_shortlist + remainder
        score_lookup = {resume_id: float(len(ordered_all) - rank_index) for rank_index, resume_id in enumerate(ordered_all)}
        query_rank_rows = pd.DataFrame({
            "job_id": job_id,
            "resume_id": ordered_all,
            "score": [score_lookup[resume_id] for resume_id in ordered_all],
        })
        query_rank_rows.to_csv(raw_rankings_path, mode="a", header=not raw_rankings_path.exists(), index=False)

        reason_rows = pd.DataFrame(
            [
                {
                    "job_id": job_id,
                    "resume_id": item["resume_id"],
                    "shortlist_rank": position,
                    "reason": item.get("reason", ""),
                    "parse_strategy": strategy,
                }
                for position, item in enumerate(parsed_items, start=1)
            ]
        )
        if not reason_rows.empty:
            reason_rows.to_csv(reasons_path, mode="a", header=not reasons_path.exists(), index=False)

        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "job_id": job_id,
                "parse_strategy": strategy,
                "candidate_ids": lexical_ids,
                "ordered_ids": ordered_shortlist,
                "latency_seconds": latency_seconds,
                "response_text": response_text,
            }, ensure_ascii=True) + "\n")

        processed_job_ids.add(job_id)
        if checkpoint_every > 0 and (len(processed_job_ids) % checkpoint_every == 0 or len(processed_job_ids) == len(target_jobs)):
            print(f"Processed {len(processed_job_ids)}/{len(target_jobs)} LLM queries")

    runtime_seconds = time.perf_counter() - start
    if not raw_rankings_path.exists():
        raise RuntimeError("No LLM rankings were written during the run.")

    scored_pairs = pd.read_csv(raw_rankings_path)
    rankings = build_ranking_results(scored_pairs, model_name=model_name, split="test")
    eval_split = split_frame.loc[split_frame["job_id"].isin(target_jobs)].copy()
    aggregate_metrics, query_metrics = evaluate_rankings(rankings, eval_split)
    save_metric_bundle(artifact_prefix, aggregate_metrics, query_metrics)
    labeled_rankings = attach_relevance_labels(rankings, eval_split)
    labeled_rankings.to_csv(METRICS_DIR / f"{artifact_stem}_rankings_labeled.csv", index=False)

    reasons = pd.read_csv(reasons_path) if reasons_path.exists() else pd.DataFrame(columns=["job_id", "resume_id", "shortlist_rank", "reason", "parse_strategy"])
    traces = pd.DataFrame(_read_jsonl(trace_path))
    if not traces.empty and "latency_seconds" in traces.columns:
        runtime_seconds = float(traces["latency_seconds"].sum())

    return {
        "rankings": rankings,
        "aggregate_metrics": aggregate_metrics,
        "query_metrics": query_metrics,
        "reasons": reasons,
        "traces": traces,
        "runtime_seconds": runtime_seconds,
        "subset_job_ids": target_jobs,
        "detected_model_id": DETECTED_LLM_MODEL_ID,
    }


def run_llm_subset_experiment(split_frame: pd.DataFrame, max_queries: int = LLM_SUBSET_QUERY_COUNT, artifact_prefix: str = "llm_test_subset100_metrics") -> dict[str, Any]:
    return run_llm_experiment(
        split_frame,
        max_queries=max_queries,
        artifact_prefix=artifact_prefix,
        reasons_filename="llm_subset_reasons.csv",
    )


def run_llm_full_test_experiment(split_frame: pd.DataFrame, max_queries: int | None = LLM_FULL_TEST_QUERY_LIMIT, artifact_prefix: str = "llm_full_test_metrics") -> dict[str, Any]:
    return run_llm_experiment(
        split_frame,
        max_queries=max_queries,
        artifact_prefix=artifact_prefix,
        reasons_filename="llm_full_test_reasons.csv",
    )

# Saved-output fallback plus the executed LLM section.

def load_saved_llm_outputs(artifact_stem: str, reasons_filename: str) -> dict[str, Any]:
    active_metrics_dir = METRICS_DIR
    active_rankings_path = active_metrics_dir / f"{artifact_stem}_rankings_labeled.csv"
    active_aggregate_path = active_metrics_dir / f"{artifact_stem}_metrics.csv"
    active_query_path = active_metrics_dir / f"{artifact_stem}_query_metrics.csv"
    active_trace_path = active_metrics_dir / f"{artifact_stem}_traces.jsonl"
    reasons_path = TABLES_DIR / reasons_filename

    if active_aggregate_path.exists() and active_query_path.exists() and active_rankings_path.exists():
        aggregate_metrics = pd.read_csv(active_aggregate_path)
        query_metrics = pd.read_csv(active_query_path)
        rankings = pd.read_csv(active_rankings_path)
        reasons = pd.read_csv(reasons_path) if reasons_path.exists() else pd.DataFrame(columns=["job_id", "resume_id", "shortlist_rank", "reason", "parse_strategy"])
        traces = pd.DataFrame(_read_jsonl(active_trace_path))
        runtime_seconds = float(traces["latency_seconds"].sum()) if not traces.empty and "latency_seconds" in traces.columns else float("nan")
        subset_job_ids = sorted(pd.Series(rankings["job_id"]).drop_duplicates().tolist())
        return {
            "rankings": rankings,
            "aggregate_metrics": aggregate_metrics,
            "query_metrics": query_metrics,
            "reasons": reasons,
            "traces": traces,
            "runtime_seconds": runtime_seconds,
            "subset_job_ids": subset_job_ids,
        }

    legacy_dir = PROJECT_ROOT / "archive" / "legacy_results" / "metrics"
    if artifact_stem == "llm_test_subset100":
        aggregate_metrics = pd.read_csv(legacy_dir / "llm_test_q100_short3_c650_550__aggregate_metrics.csv")
        query_metrics = pd.read_csv(legacy_dir / "llm_test_q100_short3_c650_550__query_metrics.csv")
        rankings = pd.read_csv(legacy_dir / "llm_test_q100_short3_c650_550__rankings_labeled.csv")
        trace_rows = _read_jsonl(legacy_dir / "llm_test_q100_short3_c650_550__llm_traces.jsonl")
        reasons = pd.read_csv(reasons_path) if reasons_path.exists() else pd.DataFrame(columns=["job_id", "resume_id", "shortlist_rank", "reason", "parse_strategy"])
        traces = pd.DataFrame(trace_rows)
        runtime_seconds = float(traces.get("latency_seconds", pd.Series(dtype=float)).sum()) if not traces.empty else float("nan")
        subset_job_ids = sorted(pd.Series(rankings["job_id"]).drop_duplicates().tolist())
        return {
            "rankings": rankings,
            "aggregate_metrics": aggregate_metrics,
            "query_metrics": query_metrics,
            "reasons": reasons,
            "traces": traces,
            "runtime_seconds": runtime_seconds,
            "subset_job_ids": subset_job_ids,
        }

    raise FileNotFoundError(f"No saved LLM outputs found for artifact stem: {artifact_stem}")
