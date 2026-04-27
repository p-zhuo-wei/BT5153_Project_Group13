"""Pure ranking metric functions: P@k, R@k, MRR, MAP, nDCG@k.

Cell-equivalent: Step 2B (cell 23).
"""

from typing import Sequence

import numpy as np


# Ranking metrics used throughout the comparison.

def precision_at_k(relevances: Sequence[int], k: int, positive_threshold: int = 1) -> float:
    top_k = list(relevances[:k])
    if not top_k:
        return 0.0
    return sum(1 for value in top_k if value >= positive_threshold) / k


def recall_at_k(relevances: Sequence[int], k: int, total_relevant: int, positive_threshold: int = 1) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(1 for value in relevances[:k] if value >= positive_threshold) / total_relevant


def reciprocal_rank(relevances: Sequence[int], positive_threshold: int = 1) -> float:
    for index, value in enumerate(relevances, start=1):
        if value >= positive_threshold:
            return 1.0 / index
    return 0.0


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    top_k = np.asarray(list(relevances[:k]), dtype=float)
    if top_k.size == 0:
        return 0.0
    gains = np.power(2.0, top_k) - 1.0
    discounts = np.log2(np.arange(2, top_k.size + 2, dtype=float))
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: Sequence[int], k: int, ideal_relevances: Sequence[int] | None = None) -> float:
    observed = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(ideal_relevances if ideal_relevances is not None else relevances, reverse=True), k)
    if ideal <= 0.0:
        return 0.0
    return observed / ideal


def average_precision(relevances: Sequence[int], total_relevant: int, positive_threshold: int = 1) -> float:
    if total_relevant <= 0:
        return 0.0
    hit_count = 0
    precision_sum = 0.0
    for rank, value in enumerate(relevances, start=1):
        if value >= positive_threshold:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / total_relevant
