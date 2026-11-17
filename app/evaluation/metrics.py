import re
from collections.abc import Sequence
from math import log2


def normalized_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def token_scores(prediction: str, reference: str) -> dict[str, float]:
    predicted_tokens = normalized_tokens(prediction)
    reference_tokens = normalized_tokens(reference)
    overlap = len(predicted_tokens & reference_tokens)
    precision = overlap / len(predicted_tokens) if predicted_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def retrieval_scores(
    ranked_keys: Sequence[tuple[str, int]],
    relevance_by_key: dict[tuple[str, int], int],
    cutoff: int,
) -> dict[str, float]:
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")

    ranked = list(ranked_keys[:cutoff])
    relevant_count = sum(relevance > 0 for relevance in relevance_by_key.values())
    relevant_ranks = [
        rank
        for rank, key in enumerate(ranked, start=1)
        if relevance_by_key.get(key, 0) > 0
    ]

    dcg = sum(
        (2 ** relevance_by_key.get(key, 0) - 1) / log2(rank + 1)
        for rank, key in enumerate(ranked, start=1)
    )
    ideal_relevances = sorted(relevance_by_key.values(), reverse=True)[:cutoff]
    ideal_dcg = sum(
        (2**relevance - 1) / log2(rank + 1)
        for rank, relevance in enumerate(ideal_relevances, start=1)
    )

    return {
        "hit_rate": float(bool(relevant_ranks)),
        "recall": len(relevant_ranks) / relevant_count if relevant_count else 0.0,
        "mrr": 1 / relevant_ranks[0] if relevant_ranks else 0.0,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def citation_scores(
    predicted: Sequence[tuple[str, int]],
    expected: Sequence[tuple[str, int]],
) -> dict[str, float]:
    predicted_set = set(predicted)
    expected_set = set(expected)
    overlap = len(predicted_set & expected_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def mean_metrics(case_results: Sequence[dict]) -> dict[str, dict[str, float]]:
    group_names = {
        group_name
        for result in case_results
        for group_name, group in result.items()
        if isinstance(group, dict)
    }
    aggregate = {}
    for group_name in sorted(group_names):
        metric_names = {
            metric_name
            for result in case_results
            for metric_name, value in result.get(group_name, {}).items()
            if isinstance(value, int | float)
        }
        group_means = {}
        for metric_name in sorted(metric_names):
            values = [
                result[group_name][metric_name]
                for result in case_results
                if isinstance(result.get(group_name, {}).get(metric_name), int | float)
            ]
            if values:
                group_means[metric_name] = sum(values) / len(values)
        aggregate[group_name] = group_means
    return aggregate
