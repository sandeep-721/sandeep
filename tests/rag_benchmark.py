import json
import time
from collections import defaultdict
from pathlib import Path

from retrieval.hybrid_search import HybridSearch


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "tests" / "data" / "rag_benchmark.json"
REPORT = ROOT / "tests" / "data" / "rag_benchmark_report.json"

DENSE_LIMIT = 100
RRF_LIMIT = 75
FINAL_LIMIT = 20


CATEGORY_MAP = {
    "facial_01": "code_symbol",
    "facial_02": "code_semantic",
    "facial_03": "code_semantic",
    "facial_04": "documentation",
    "facial_05": "documentation",
    "facial_06": "configuration",
    "painter_01": "documentation_workflow",
    "painter_02": "history",
    "painter_03": "documentation_workflow",
    "painter_04": "history",
    "painter_05": "code_async",
    "painter_06": "code_test",
    "rag_01": "configuration",
    "rag_02": "configuration",
    "rag_03": "code_architecture",
    "rag_04": "code_retrieval",
    "rag_05": "code_embedding",
    "rag_06": "code_metadata",
}


def get_source(result):
    payload = result.get("payload", {}) or {}
    return payload.get("source", "")


def first_gold_rank(results, gold_sources):
    gold = set(gold_sources)

    for rank, result in enumerate(results, start=1):
        if get_source(result) in gold:
            return rank

    return None


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    return 1.0 / rank


def recall_at(rank, k):
    if rank is None:
        return 0.0

    return 1.0 if rank <= k else 0.0


def print_regression_diagnostics(case, rrf, final, rrf_rank, final_rank):
    if (
        rrf_rank is not None
        and final_rank is not None
        and final_rank <= rrf_rank
    ):
        return

    print("\n--- RERANK REGRESSION DIAGNOSTIC ---")
    print(
        f"RRF rank={rrf_rank if rrf_rank is not None else 'MISS'} "
        f"FINAL rank={final_rank if final_rank is not None else 'MISS'}"
    )

    gold_sources = set(case.get("gold_sources", []))

    for rank, document in enumerate(final[:10], start=1):
        payload = document.get("payload", {}) or {}
        source = payload.get("source", "")
        symbol = payload.get("symbol", "")
        chunk_type = payload.get("chunk_type", "")

        gold_tag = "GOLD" if source in gold_sources else "    "

        print(
            f"{gold_tag} "
            f"rank={rank:02d} "
            f"final={document.get('final_rerank_score', 0.0):.4f} "
            f"cross={document.get('rerank_normalized_score', 0.0):.4f} "
            f"lex={document.get('lexical_score', 0.0):.4f} "
            f"answer={document.get('answer_evidence_score', 0.0):.4f} "
            f"rrf={document.get('rrf_preservation_score', 0.0):.4f} "
            f"auth={document.get('source_authority', 0.0):.4f} "
            f"tier={document.get('structural_tier', 0)} "
            f"source={source} "
            f"symbol={symbol} "
            f"chunk={chunk_type}"
        )

def summarize(rows):
    if not rows:
        return {
            "count": 0,
            "mrr": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "recall_at_20": 0.0,
            "misses": 0,
        }

    ranks = [row["rank"] for row in rows]
    count = len(ranks)

    return {
        "count": count,
        "mrr": sum(
            reciprocal_rank(rank)
            for rank in ranks
        ) / count,
        "recall_at_5": sum(
            recall_at(rank, 5)
            for rank in ranks
        ) / count,
        "recall_at_10": sum(
            recall_at(rank, 10)
            for rank in ranks
        ) / count,
        "recall_at_20": sum(
            recall_at(rank, 20)
            for rank in ranks
        ) / count,
        "misses": sum(
            1
            for rank in ranks
            if rank is None
        ),
    }


def main():
    cases = json.loads(
        DATASET.read_text(
            encoding="utf-8-sig"
        )
    )

    search = HybridSearch()

    stage_rows = defaultdict(list)
    project_rows = defaultdict(
        lambda: defaultdict(list)
    )
    category_rows = defaultdict(
        lambda: defaultdict(list)
    )

    case_reports = []

    try:
        for case in cases:
            case_id = case["id"]
            project = case["project"]
            category = CATEGORY_MAP.get(
                case_id,
                "uncategorized",
            )

            print("\n" + "=" * 100)
            print(
                f"{case_id} | "
                f"{project} | "
                f"{category}"
            )
            print(case["query"])

            timings = {}

            started = time.perf_counter()
            dense = search.dense.search(
                query=case["query"],
                limit=DENSE_LIMIT,
                project=project,
            )
            timings["dense_ms"] = (
                time.perf_counter() - started
            ) * 1000

            started = time.perf_counter()
            sparse = search.sparse.search(
                query=case["query"],
                limit=DENSE_LIMIT,
                project=project,
            )
            timings["sparse_ms"] = (
                time.perf_counter() - started
            ) * 1000

            started = time.perf_counter()
            query_intent = (
                search.dense._detect_query_intent(
                    case["query"]
                )
            )

            rrf = search._fuse_results(
                dense_results=dense,
                sparse_results=sparse,
                candidate_limit=RRF_LIMIT,
                query_intent=query_intent,
            )
            timings["rrf_ms"] = (
                time.perf_counter() - started
            ) * 1000

            started = time.perf_counter()
            final = search.reranker.rerank(
                query=case["query"],
                documents=rrf,
                limit=FINAL_LIMIT,
            )
            timings["reranker_ms"] = (
                time.perf_counter() - started
            ) * 1000

            results = {
                "dense": dense,
                "sparse": sparse,
                "rrf": rrf,
                "final": final,
            }

            ranks = {}

            for stage_name, stage_results in results.items():
                rank = first_gold_rank(
                    stage_results,
                    case["gold_sources"],
                )

                ranks[stage_name] = rank

                stage_rows[stage_name].append(
                    {
                        "case_id": case_id,
                        "project": project,
                        "category": category,
                        "rank": rank,
                    }
                )

                project_rows[project][stage_name].append(
                    {
                        "case_id": case_id,
                        "category": category,
                        "rank": rank,
                    }
                )

                category_rows[category][stage_name].append(
                    {
                        "case_id": case_id,
                        "project": project,
                        "rank": rank,
                    }
                )

                print(
                    f"{stage_name:8} "
                    f"rank="
                    f"{rank if rank is not None else 'MISS'}"
                )

            print(
                "timing   "
                f"dense={timings['dense_ms']:.1f}ms "
                f"sparse={timings['sparse_ms']:.1f}ms "
                f"rrf={timings['rrf_ms']:.1f}ms "
                f"rerank={timings['reranker_ms']:.1f}ms"
            )

            print_regression_diagnostics(
                case=case,
                rrf=rrf,
                final=final,
                rrf_rank=ranks["rrf"],
                final_rank=ranks["final"],
            )

            case_reports.append(
                {
                    "id": case_id,
                    "project": project,
                    "category": category,
                    "query": case["query"],
                    "gold_sources": case["gold_sources"],
                    "ranks": ranks,
                    "timings_ms": timings,
                }
            )

        overall = {
            stage: summarize(rows)
            for stage, rows in stage_rows.items()
        }

        by_project = {}

        for project, stages in project_rows.items():
            by_project[project] = {
                stage: summarize(rows)
                for stage, rows in stages.items()
            }

        by_category = {}

        for category, stages in category_rows.items():
            by_category[category] = {
                stage: summarize(rows)
                for stage, rows in stages.items()
            }

        report = {
            "dataset": str(DATASET),
            "case_count": len(cases),
            "limits": {
                "dense": DENSE_LIMIT,
                "rrf": RRF_LIMIT,
                "final": FINAL_LIMIT,
            },
            "overall": overall,
            "by_project": by_project,
            "by_category": by_category,
            "cases": case_reports,
        }

        REPORT.write_text(
            json.dumps(
                report,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("\n" + "=" * 100)
        print("OVERALL")
        print("=" * 100)

        for stage, metrics in overall.items():
            print(f"\n{stage.upper()}")
            print(
                f"  Cases      : {metrics['count']}"
            )
            print(
                f"  MRR        : {metrics['mrr']:.4f}"
            )
            print(
                f"  Recall@5   : {metrics['recall_at_5']:.4f}"
            )
            print(
                f"  Recall@10  : {metrics['recall_at_10']:.4f}"
            )
            print(
                f"  Recall@20  : {metrics['recall_at_20']:.4f}"
            )
            print(
                f"  Misses     : {metrics['misses']}"
            )

        print("\n" + "=" * 100)
        print("BY PROJECT")
        print("=" * 100)

        for project, stages in sorted(
            by_project.items()
        ):
            print(f"\n[{project}]")

            for stage, metrics in stages.items():
                print(
                    f"  {stage:8} "
                    f"MRR={metrics['mrr']:.4f} "
                    f"R@5={metrics['recall_at_5']:.4f} "
                    f"R@10={metrics['recall_at_10']:.4f}"
                )

        print("\n" + "=" * 100)
        print("BY CATEGORY")
        print("=" * 100)

        for category, stages in sorted(
            by_category.items()
        ):
            print(f"\n[{category}]")

            for stage, metrics in stages.items():
                print(
                    f"  {stage:8} "
                    f"MRR={metrics['mrr']:.4f} "
                    f"R@5={metrics['recall_at_5']:.4f} "
                    f"R@10={metrics['recall_at_10']:.4f}"
                )

        print("\nReport written to:")
        print(REPORT)

    finally:
        search.close()


if __name__ == "__main__":
    main()