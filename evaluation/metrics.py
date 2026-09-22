from __future__ import annotations


def _source_name(source: str) -> str:
    """
    Return the filename portion of a source path.
    """
    source = source.replace("\\", "/")
    return source.rsplit("/", 1)[-1]


def source_matches(
    actual_source: str | None,
    expected_sources: tuple[str, ...],
) -> bool:
    if not actual_source:
        return False

    actual_name = _source_name(actual_source)

    return any(
        actual_name == expected
        for expected in expected_sources
    )


def symbol_matches(
    payload: dict,
    expected_symbols: tuple[str, ...],
) -> bool:
    if not expected_symbols:
        return False

    symbol = payload.get("symbol")

    if symbol in expected_symbols:
        return True

    class_name = payload.get("class_name")

    if class_name in expected_symbols:
        return True

    member_name = payload.get("member_name")

    if member_name in expected_symbols:
        return True

    return False


def reciprocal_rank(
    results: list[dict],
    expected_sources: tuple[str, ...],
) -> float:
    """
    Mean reciprocal rank for the first relevant source.

    Returns 0 when no relevant source is retrieved.
    """
    for rank, result in enumerate(results, start=1):
        payload = result.get("payload", {})

        if source_matches(
            payload.get("source"),
            expected_sources,
        ):
            return 1.0 / rank

    return 0.0


def hit_rate(
    results: list[dict],
    expected_sources: tuple[str, ...],
) -> float:
    """
    Returns 1 when at least one expected source is retrieved,
    otherwise 0.
    """
    for result in results:
        payload = result.get("payload", {})

        if source_matches(
            payload.get("source"),
            expected_sources,
        ):
            return 1.0

    return 0.0


def symbol_hit_rate(
    results: list[dict],
    expected_symbols: tuple[str, ...],
) -> float:
    """
    Returns 1 when at least one expected symbol is retrieved.
    """
    if not expected_symbols:
        return 0.0

    for result in results:
        payload = result.get("payload", {})

        if symbol_matches(
            payload,
            expected_symbols,
        ):
            return 1.0

    return 0.0


def precision_at_k(
    results: list[dict],
    expected_sources: tuple[str, ...],
    k: int,
) -> float:
    """
    Fraction of the first k results belonging to an expected source.
    """
    if k <= 0:
        return 0.0

    selected = results[:k]

    if not selected:
        return 0.0

    relevant = 0

    for result in selected:
        payload = result.get("payload", {})

        if source_matches(
            payload.get("source"),
            expected_sources,
        ):
            relevant += 1

    return relevant / len(selected)


def reciprocal_rank_by_symbol(
    results: list[dict],
    expected_symbols: tuple[str, ...],
) -> float:
    """
    Reciprocal rank for the first result containing an expected symbol.
    """
    if not expected_symbols:
        return 0.0

    for rank, result in enumerate(results, start=1):
        payload = result.get("payload", {})

        if symbol_matches(
            payload,
            expected_symbols,
        ):
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    results: list[dict],
    expected_sources: tuple[str, ...],
    expected_symbols: tuple[str, ...] = (),
) -> dict:
    """
    Evaluate one retrieval stage using the same metrics.

    This allows dense, sparse, hybrid, and reranked results
    to be compared using identical measurements.
    """
    return {
        "hit_rate": hit_rate(
            results,
            expected_sources,
        ),
        "mrr": reciprocal_rank(
            results,
            expected_sources,
        ),
        "precision_at_5": precision_at_k(
            results,
            expected_sources,
            5,
        ),
        "symbol_hit_rate": symbol_hit_rate(
            results,
            expected_symbols,
        ),
        "symbol_mrr": reciprocal_rank_by_symbol(
            results,
            expected_symbols,
        ),
    }


def average_metrics(
    evaluations: list[dict],
) -> dict:
    """
    Calculate the arithmetic mean for each evaluation metric.
    """
    if not evaluations:
        return {}

    metric_names = (
        "hit_rate",
        "mrr",
        "precision_at_5",
        "symbol_hit_rate",
        "symbol_mrr",
    )

    return {
        metric: (
            sum(
                evaluation.get(
                    metric,
                    0.0,
                )
                for evaluation in evaluations
            )
            / len(evaluations)
        )
        for metric in metric_names
    }