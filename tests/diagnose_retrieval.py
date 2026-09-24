from __future__ import annotations

import argparse

from retrieval.hybrid_search import HybridSearch


RETRIEVAL_LIMIT = 100
RRF_CANDIDATE_LIMIT = 75
RERANK_LIMIT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose a query through the production HybridSearch pipeline."
    )

    parser.add_argument(
        "query",
        help="Query to diagnose.",
    )

    parser.add_argument(
        "--target-id",
        default=None,
        help="Optional Qdrant point ID of the expected target document.",
    )

    parser.add_argument(
        "--target-path",
        default=None,
        help="Optional document path of the expected target. Used when --target-id is omitted.",
    )

    return parser.parse_args()


def get_payload(result):
    if isinstance(result, dict):
        return result.get("payload", {}) or {}
    return result.payload or {}


def get_id(result):
    if isinstance(result, dict):
        return result.get("id")
    return result.id


def get_score(result):
    if isinstance(result, dict):
        return result.get("score")
    return result.score


def get_path(result):
    payload = get_payload(result)
    return payload.get("path", "")


def matches_target(result, target_id=None, target_path=None):
    if target_id is not None and str(get_id(result)) == str(target_id):
        return True

    if target_path is not None:
        result_path = get_path(result)
        if result_path == target_path:
            return True

    return False


def find_target(results, target_id=None, target_path=None):
    if target_id is None and target_path is None:
        return None, None

    for rank, result in enumerate(results, start=1):
        if matches_target(
            result,
            target_id=target_id,
            target_path=target_path,
        ):
            return rank, result

    return None, None


def print_target(
    stage,
    results,
    target_id=None,
    target_path=None,
):
    rank, result = find_target(
        results,
        target_id=target_id,
        target_path=target_path,
    )

    print(f"\n{stage}:")

    if target_id is None and target_path is None:
        print("  No target supplied; showing candidate count only.")
        return

    if result is None:
        print("  NOT FOUND")
        return

    payload = get_payload(result)

    print(f"  Rank        : {rank}")
    print(f"  ID          : {get_id(result)}")
    print(f"  Path        : {get_path(result)}")
    print(f"  Score       : {get_score(result)}")

    if isinstance(result, dict):
        if "dense_rank" in result:
            print(f"  Dense rank  : {result.get('dense_rank')}")
        if "sparse_rank" in result:
            print(f"  Sparse rank : {result.get('sparse_rank')}")
        if "dense_rrf" in result:
            print(f"  Dense RRF   : {result.get('dense_rrf')}")
        if "sparse_rrf" in result:
            print(f"  Sparse RRF  : {result.get('sparse_rrf')}")

    text = payload.get("text", "")

    if text:
        print("\n  Text:")
        print("  " + text[:1000].replace("\n", "\n  "))


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def main():
    args = parse_args()

    query = args.query
    target_id = args.target_id
    target_path = args.target_path

    print("=" * 100)
    print("PRODUCTION-EXACT RRF -> RERANKER DIAGNOSTIC")
    print("=" * 100)

    print("\nQuery:")
    print(query)

    print("\nTarget:")
    if target_id:
        print(f"ID   : {target_id}")
    if target_path:
        print(f"Path : {target_path}")
    if not target_id and not target_path:
        print("No target supplied.")

    search = HybridSearch()

    # ------------------------------------------------------------------
    # 1. DENSE
    # ------------------------------------------------------------------
    print_header("1. DENSE RESULTS")

    dense_results = search.dense.search(
        query,
        limit=RETRIEVAL_LIMIT,
    )

    print(f"Total dense candidates: {len(dense_results)}")

    print_target(
        "DENSE TARGET",
        dense_results,
        target_id=target_id,
        target_path=target_path,
    )

    # ------------------------------------------------------------------
    # 2. SPARSE
    # ------------------------------------------------------------------
    print_header("2. SPARSE RESULTS")

    sparse_results = search.sparse.search(
        query,
        limit=RETRIEVAL_LIMIT,
    )

    print(f"Total sparse candidates: {len(sparse_results)}")

    print_target(
        "SPARSE TARGET",
        sparse_results,
        target_id=target_id,
        target_path=target_path,
    )

    # ------------------------------------------------------------------
    # 3. RRF
    # ------------------------------------------------------------------
    print_header("3. RRF CANDIDATES")

    rrf_results = search._fuse_results(
        dense_results=dense_results,
        sparse_results=sparse_results,
        candidate_limit=RRF_CANDIDATE_LIMIT,
    )

    print(f"Total RRF candidates: {len(rrf_results)}")

    print_target(
        "RRF TARGET",
        rrf_results,
        target_id=target_id,
        target_path=target_path,
    )

    # ------------------------------------------------------------------
    # 4. EXACT PRODUCTION RERANKER INPUT
    # ------------------------------------------------------------------
    print_header("4. EXACT PRODUCTION RERANKER INPUT")

    # Production HybridSearch sends ALL RRF candidates to the reranker.
    # Do NOT slice this to [:20].
    reranker_input = rrf_results

    print(f"Reranker input candidates: {len(reranker_input)}")

    rank, target = find_target(
        reranker_input,
        target_id=target_id,
        target_path=target_path,
    )

    if target_id is None and target_path is None:
        print("\nNo target supplied; skipping target text inspection.")
    elif target is None:
        print("\nTARGET IS NOT IN RERANKER INPUT.")
    else:
        payload = get_payload(target)

        print("\nTARGET IS IN RERANKER INPUT.")
        print(f"  RRF rank: {rank}")
        print(f"  ID      : {get_id(target)}")
        print(f"  Path    : {get_path(target)}")

        print("\n  EXACT TEXT SENT TO RERANKER:")
        print("-" * 100)
        print(payload.get("text", ""))
        print("-" * 100)

    # ------------------------------------------------------------------
    # 5. PRODUCTION RERANKER
    # ------------------------------------------------------------------
    print_header("5. PRODUCTION QWEN3-RERANKER")

    reranked_results = search.reranker.rerank(
        query=query,
        documents=reranker_input,
        limit=RERANK_LIMIT,
    )

    print(f"Reranked candidates: {len(reranked_results)}")

    print_target(
        "RERANKED TARGET",
        reranked_results,
        target_id=target_id,
        target_path=target_path,
    )

    # ------------------------------------------------------------------
    # 6. FINAL PRODUCTION HYBRID SEARCH
    # ------------------------------------------------------------------
    print_header("6. FINAL PRODUCTION HYBRID SEARCH")

    final_results = search.search(
        query,
        limit=RERANK_LIMIT,
    )

    print(f"Final results: {len(final_results)}")

    print_target(
        "FINAL TARGET",
        final_results,
        target_id=target_id,
        target_path=target_path,
    )

    # ------------------------------------------------------------------
    # 7. SUMMARY
    # ------------------------------------------------------------------
    print_header("7. SUMMARY")

    if target_id is None and target_path is None:
        print("No target supplied.")
        print("Supply --target-id or --target-path for rank tracking.")
        return

    dense_rank, _ = find_target(
        dense_results,
        target_id=target_id,
        target_path=target_path,
    )

    sparse_rank, _ = find_target(
        sparse_results,
        target_id=target_id,
        target_path=target_path,
    )

    rrf_rank, rrf_target = find_target(
        rrf_results,
        target_id=target_id,
        target_path=target_path,
    )

    rerank_rank, _ = find_target(
        reranked_results,
        target_id=target_id,
        target_path=target_path,
    )

    final_rank, _ = find_target(
        final_results,
        target_id=target_id,
        target_path=target_path,
    )

    print(
        f"Dense      : "
        f"{'FOUND #' + str(dense_rank) if dense_rank else 'NOT FOUND'}"
    )

    print(
        f"Sparse     : "
        f"{'FOUND #' + str(sparse_rank) if sparse_rank else 'NOT FOUND'}"
    )

    print(
        f"RRF        : "
        f"{'FOUND #' + str(rrf_rank) if rrf_rank else 'NOT FOUND'}"
    )

    print(
        f"Reranker   : "
        f"{'FOUND #' + str(rerank_rank) if rerank_rank else 'NOT FOUND'}"
    )

    print(
        f"Final      : "
        f"{'FOUND #' + str(final_rank) if final_rank else 'NOT FOUND'}"
    )

    if rrf_target:
        print("\nRRF details:")

        if isinstance(rrf_target, dict):
            print(f"  Dense rank  : {rrf_target.get('dense_rank')}")
            print(f"  Sparse rank : {rrf_target.get('sparse_rank')}")
            print(f"  RRF score   : {rrf_target.get('score')}")

    print("\nProduction configuration:")
    print(f"  Dense limit : {RETRIEVAL_LIMIT}")
    print(f"  Sparse limit: {RETRIEVAL_LIMIT}")
    print(f"  RRF limit   : {RRF_CANDIDATE_LIMIT}")
    print(f"  Rerank limit: {RERANK_LIMIT}")

    print("\nPipeline:")
    print("Dense -> Sparse -> RRF -> Qwen3-Reranker -> Final ranking")

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
