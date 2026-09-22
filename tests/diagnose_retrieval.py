from retrieval.hybrid_search import HybridSearch


QUERY = "What device is configured for runtime inference?"
TARGET_ID = "9efb9e92-cc69-4a9d-be44-17596f4d1450"

RETRIEVAL_LIMIT = 100
RRF_CANDIDATE_LIMIT = 75
RERANK_LIMIT = 20


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


def find_target(results):
    for rank, result in enumerate(results, start=1):
        if get_id(result) == TARGET_ID:
            return rank, result
    return None, None


def print_target(stage, results):
    rank, result = find_target(results)

    print(f"\n{stage}:")

    if result is None:
        print("  NOT FOUND")
        return

    payload = get_payload(result)

    print(f"  Rank        : {rank}")
    print(f"  ID          : {get_id(result)}")
    print(f"  Path        : {get_path(result)}")
    print(f"  Score       : {get_score(result)}")

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


print("=" * 100)
print("PRODUCTION-EXACT RRF -> RERANKER DIAGNOSTIC")
print("=" * 100)

print(f"Query:")
print(QUERY)

print("\nTarget document:")
print(TARGET_ID)
print("config/settings.py")


search = HybridSearch()

try:
    # ------------------------------------------------------------------
    # 1. DENSE
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("1. DENSE RESULTS")
    print("=" * 100)

    dense_results = search.dense.search(
        QUERY,
        limit=RETRIEVAL_LIMIT,
    )

    print(f"Total dense candidates: {len(dense_results)}")
    print_target("DENSE TARGET", dense_results)

    # ------------------------------------------------------------------
    # 2. SPARSE
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("2. SPARSE RESULTS")
    print("=" * 100)

    sparse_results = search.sparse.search(
        QUERY,
        limit=RETRIEVAL_LIMIT,
    )

    print(f"Total sparse candidates: {len(sparse_results)}")
    print_target("SPARSE TARGET", sparse_results)

    # ------------------------------------------------------------------
    # 3. RRF
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("3. RRF CANDIDATES")
    print("=" * 100)

    rrf_results = search._fuse_results(
        dense_results=dense_results,
        sparse_results=sparse_results,
        candidate_limit=RRF_CANDIDATE_LIMIT,
    )

    print(f"Total RRF candidates: {len(rrf_results)}")
    print_target("RRF TARGET", rrf_results)

    # ------------------------------------------------------------------
    # 4. EXACT PRODUCTION RERANKER INPUT
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("4. EXACT PRODUCTION RERANKER INPUT")
    print("=" * 100)

    # IMPORTANT:
    # Production HybridSearch sends ALL RRF candidates to the reranker.
    # Do NOT slice this to [:20].
    reranker_input = rrf_results

    print(f"Reranker input candidates: {len(reranker_input)}")

    rank, target = find_target(reranker_input)

    if target is None:
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
    print("\n" + "=" * 100)
    print("5. PRODUCTION QWEN3-RERANKER")
    print("=" * 100)

    reranked_results = search.reranker.rerank(
        query=QUERY,
        documents=reranker_input,
        limit=RERANK_LIMIT,
    )

    print(f"Reranked candidates: {len(reranked_results)}")
    print_target("RERANKED TARGET", reranked_results)

    # ------------------------------------------------------------------
    # 6. FINAL PRODUCTION HYBRID SEARCH
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("6. FINAL PRODUCTION HYBRID SEARCH")
    print("=" * 100)

    final_results = search.search(
        QUERY,
        limit=RERANK_LIMIT,
    )

    print(f"Final results: {len(final_results)}")
    print_target("FINAL TARGET", final_results)

    # ------------------------------------------------------------------
    # 7. SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("7. SUMMARY")
    print("=" * 100)

    dense_rank, _ = find_target(dense_results)
    sparse_rank, _ = find_target(sparse_results)
    rrf_rank, rrf_target = find_target(rrf_results)
    rerank_rank, _ = find_target(reranked_results)
    final_rank, _ = find_target(final_results)

    print(f"Dense      : {'FOUND #' + str(dense_rank) if dense_rank else 'NOT FOUND'}")
    print(f"Sparse     : {'FOUND #' + str(sparse_rank) if sparse_rank else 'NOT FOUND'}")
    print(f"RRF        : {'FOUND #' + str(rrf_rank) if rrf_rank else 'NOT FOUND'}")
    print(f"Reranker   : {'FOUND #' + str(rerank_rank) if rerank_rank else 'NOT FOUND'}")
    print(f"Final      : {'FOUND #' + str(final_rank) if final_rank else 'NOT FOUND'}")

    if rrf_target:
        print("\nRRF details:")
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

finally:
    search.close()

print("\nDiagnostic complete.")