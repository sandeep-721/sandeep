from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.cases import EVALUATION_CASES
from evaluation.metrics import average_metrics, evaluate_retrieval
from retrieval.hybrid_search import HybridSearch


def display_name(source: str | None) -> str:
    if not source:
        return "<unknown>"

    return source.replace("\\", "/").rsplit("/", 1)[-1]


def result_source(result: dict) -> str:
    payload = result.get("payload", {})

    return display_name(
        payload.get("source")
        or payload.get("path")
    )


def result_symbol(result: dict) -> str:
    payload = result.get("payload", {})

    return (
        payload.get("symbol")
        or payload.get("member_name")
        or payload.get("class_name")
        or "<none>"
    )


def result_score(result: dict) -> str:
    score_fields = (
        "final_rerank_score",
        "hybrid_score",
        "rrf_score",
        "score",
        "semantic_score",
        "bm25_score",
    )

    for field in score_fields:
        value = result.get(field)

        if value is not None:
            try:
                return f"{float(value):.4f}"
            except (TypeError, ValueError):
                return str(value)

    return "-"


def find_expected_positions(
    results: list[dict],
    expected_sources: tuple[str, ...],
    expected_symbols: tuple[str, ...],
):
    source_positions = []
    symbol_positions = []

    expected_source_names = {
        source.replace("\\", "/").rsplit("/", 1)[-1].lower()
        for source in expected_sources
    }

    expected_symbol_names = {
        symbol.lower()
        for symbol in expected_symbols
    }

    for rank, result in enumerate(results, start=1):
        source = result_source(result).lower()
        symbol = result_symbol(result).lower()

        if source in expected_source_names:
            source_positions.append(rank)

        if symbol in expected_symbol_names:
            symbol_positions.append(rank)

    return source_positions, symbol_positions


def evaluate_case(
    hybrid: HybridSearch,
    case,
    result_limit: int = 10,
    candidate_limit: int = 50,
):
    dense_results = hybrid.dense.search(
        query=case.query,
        limit=candidate_limit,
        software=case.software,
        project=case.project,
    )

    sparse_results = hybrid.sparse.search(
        query=case.query,
        limit=candidate_limit,
        software=case.software,
        project=case.project,
    )

    rrf_results = hybrid._fuse_results(
        dense_results=dense_results,
        sparse_results=sparse_results,
        candidate_limit=candidate_limit,
    )

    reranked_results = hybrid.reranker.rerank(
        query=case.query,
        documents=rrf_results,
        limit=result_limit,
    )

    return {
        "name": case.name,
        "query": case.query,
        "expected_sources": case.expected_sources,
        "expected_symbols": case.expected_symbols,
        "dense": {
            "results": dense_results[:result_limit],
            "candidates": dense_results,
            **evaluate_retrieval(
                dense_results[:result_limit],
                case.expected_sources,
                case.expected_symbols,
            ),
        },
        "sparse": {
            "results": sparse_results[:result_limit],
            "candidates": sparse_results,
            **evaluate_retrieval(
                sparse_results[:result_limit],
                case.expected_sources,
                case.expected_symbols,
            ),
        },
        "rrf": {
            "results": rrf_results[:result_limit],
            "candidates": rrf_results,
            **evaluate_retrieval(
                rrf_results[:result_limit],
                case.expected_sources,
                case.expected_symbols,
            ),
        },
        "reranked": {
            "results": reranked_results,
            "candidates": reranked_results,
            **evaluate_retrieval(
                reranked_results,
                case.expected_sources,
                case.expected_symbols,
            ),
        },
    }


def print_stage_metrics(
    stage_name: str,
    metrics: dict,
):
    print(
        f"{stage_name:<12}"
        f"Hit@10={metrics['hit_rate']:.3f}  "
        f"MRR={metrics['mrr']:.3f}  "
        f"P@5={metrics['precision_at_5']:.3f}  "
        f"Symbol={metrics['symbol_hit_rate']:.3f}"
    )


def print_results(
    stage_name: str,
    results: list[dict],
):
    print()
    print(f"--- {stage_name.upper()} TOP RESULTS ---")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        source = result_source(result)
        symbol = result_symbol(result)
        score = result_score(result)

        print(
            f"{rank:2d}. {source:<45} score={score}"
        )

        print(
            f"    symbol={symbol}"
        )


def print_expected_diagnostics(
    evaluation,
):
    print()
    print("--- EXPECTED RESULT TRACE ---")

    expected_sources = evaluation["expected_sources"]
    expected_symbols = evaluation["expected_symbols"]

    if expected_sources:
        print(
            "Expected sources: "
            + ", ".join(expected_sources)
        )

    if expected_symbols:
        print(
            "Expected symbols: "
            + ", ".join(expected_symbols)
        )

    for stage in (
        "dense",
        "sparse",
        "rrf",
        "reranked",
    ):
        stage_data = evaluation[stage]

        source_positions, symbol_positions = (
            find_expected_positions(
                stage_data["candidates"],
                expected_sources,
                expected_symbols,
            )
        )

        source_text = (
            ", ".join(
                str(position)
                for position in source_positions
            )
            if source_positions
            else "NOT FOUND"
        )

        symbol_text = (
            ", ".join(
                str(position)
                for position in symbol_positions
            )
            if symbol_positions
            else "NOT FOUND"
        )

        print(
            f"{stage.upper():<10}"
            f"source={source_text:<15}"
            f"symbol={symbol_text}"
        )


def print_case_evaluation(
    evaluation,
):
    print()
    print("=" * 72)
    print(
        f"CASE: {evaluation['name']}"
    )
    print(
        f"QUERY: {evaluation['query']}"
    )
    print("-" * 72)

    print_stage_metrics(
        "Dense",
        evaluation["dense"],
    )

    print_stage_metrics(
        "Sparse",
        evaluation["sparse"],
    )

    print_stage_metrics(
        "RRF",
        evaluation["rrf"],
    )

    print_stage_metrics(
        "Reranked",
        evaluation["reranked"],
    )

    print_expected_diagnostics(
        evaluation
    )

    print_results(
        "Dense",
        evaluation["dense"]["results"],
    )

    print_results(
        "Sparse",
        evaluation["sparse"]["results"],
    )

    print_results(
        "RRF",
        evaluation["rrf"]["results"],
    )

    print_results(
        "Reranked",
        evaluation["reranked"]["results"],
    )


def print_summary(
    evaluations,
):
    stages = (
        "dense",
        "sparse",
        "rrf",
        "reranked",
    )

    print()
    print("=" * 72)
    print("STAGE COMPARISON SUMMARY")
    print("=" * 72)

    for stage in stages:
        metrics = average_metrics(
            [
                evaluation[stage]
                for evaluation in evaluations
            ]
        )

        print_stage_metrics(
            stage.capitalize(),
            metrics,
        )

    print("=" * 72)


def select_diagnostic_cases():
    diagnostic_names = {
        "project_progress",
        "rag_hybrid_search",
        "human_language_detection",
    }

    return [
        case
        for case in EVALUATION_CASES
        if case.name in diagnostic_names
    ]


def main():
    diagnostic_cases = select_diagnostic_cases()

    print(
        "Starting Universal RAG retrieval "
        "stage diagnostics..."
    )

    print()
    print("Diagnostic cases:")

    for case in diagnostic_cases:
        print(
            f"  - {case.name}: {case.query}"
        )

    if not diagnostic_cases:
        raise RuntimeError(
            "None of the requested diagnostic cases "
            "were found in EVALUATION_CASES."
        )

    hybrid = HybridSearch()

    try:
        evaluations = []

        for case in diagnostic_cases:
            evaluation = evaluate_case(
                hybrid,
                case,
            )

            evaluations.append(
                evaluation
            )

            print_case_evaluation(
                evaluation
            )

        print_summary(
            evaluations
        )

    finally:
        hybrid.close()


if __name__ == "__main__":
    main()