from retrieval.hybrid_search import HybridSearch

h = HybridSearch()

results = h.search(
    "How does CharacterFacialProfile manage facial control mappings and find bones?",
    limit=30,
    candidate_limit=30,
    project="Facial_Test",
)

for i, result in enumerate(results, 1):
    payload = result.get("payload", {})

    print(
        f"E{i}: "
        f"class={payload.get('class_name')} "
        f"member={payload.get('member_name')} "
        f"chunk={payload.get('chunk_index')} "
        f"rerank={result.get('rerank_score')} "
        f"boost={result.get('structural_boost')} "
        f"final={result.get('final_rerank_score')}"
    )

h.close()
