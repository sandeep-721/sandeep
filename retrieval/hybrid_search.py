from retrieval.search import SemanticSearch
from retrieval.vector_store import VectorStore
from retrieval.query_plan import QueryPlan, QueryPlanner
from retrieval.sparse_search import SparseSearch
from reranking.reranker import Reranker


class HybridSearch:
    """
    Hybrid dense + sparse retrieval using standard
    Reciprocal Rank Fusion (RRF).

    Pipeline:

        Dense retrieval  -> 100
        Sparse retrieval -> 100
        RRF candidates   -> 75
        Reranker output  -> caller limit

    Retrieval depth and RRF candidate depth are intentionally
    separate so relevant documents are not discarded before
    reranking.
    """

    DEFAULT_RETRIEVAL_LIMIT = 100
    DEFAULT_RRF_CANDIDATE_LIMIT = 75
    DEFAULT_CONTEXT_WINDOW = 1

    def __init__(self):
        self.dense = SemanticSearch()
        self.sparse = SparseSearch()
        self.reranker = Reranker()

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    @staticmethod
    def _retrieval_weights(query_intent: str):
        return QueryPlanner.RETRIEVAL_WEIGHTS.get(
            query_intent,
            QueryPlanner.RETRIEVAL_WEIGHTS["general"],
        )

    @staticmethod
    def _get_result_id(result):
        if isinstance(result, dict):
            return result.get("id")

        return result.id

    @staticmethod
    def _get_result_payload(result):
        if isinstance(result, dict):
            return result.get("payload", {})

        return result.payload

    @classmethod
    def _add_result(
        cls,
        fused,
        result,
        rank,
        retriever,
        weight=1.0,
    ):
        point_id = cls._get_result_id(result)

        if point_id is None:
            return

        payload = (
            cls._get_result_payload(result)
            or {}
        )

        if point_id not in fused:
            fused[point_id] = {
                "id": point_id,
                "payload": payload,
                "score": 0.0,
                "dense_rank": None,
                "sparse_rank": None,
                "dense_rrf": 0.0,
                "sparse_rrf": 0.0,
            }

        rrf_score = (
            cls._rrf_score(rank)
            * weight
        )

        fused[point_id]["score"] += rrf_score

        if retriever == "dense":
            fused[point_id]["dense_rank"] = rank
            fused[point_id]["dense_rrf"] = rrf_score

        elif retriever == "sparse":
            fused[point_id]["sparse_rank"] = rank
            fused[point_id]["sparse_rrf"] = rrf_score

    @classmethod
    def _fuse_results(
        cls,
        dense_results,
        sparse_results,
        candidate_limit=DEFAULT_RRF_CANDIDATE_LIMIT,
        query_intent="general",
    ):
        """
        Query-adaptive chunk-level RRF.

        The fusion remains transparent RRF, but dense/sparse
        contributions are slightly reweighted by query intent.
        """

        fused = {}

        dense_weight, sparse_weight = (
            cls._retrieval_weights(query_intent)
        )

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            cls._add_result(
                fused=fused,
                result=result,
                rank=rank,
                retriever="dense",
                weight=dense_weight,
            )

        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):
            cls._add_result(
                fused=fused,
                result=result,
                rank=rank,
                retriever="sparse",
                weight=sparse_weight,
            )

        if not fused:
            return []

        ranked = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        return ranked[:candidate_limit]

    def search(
        self,
        query: str,
        limit: int = 10,
        retrieval_limit: int = DEFAULT_RETRIEVAL_LIMIT,
        candidate_limit: int = DEFAULT_RRF_CANDIDATE_LIMIT,
        software: str | None = None,
        software_version: str | None = None,
        project: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
        human_language: str | None = None,
        query_plan: QueryPlan | None = None,
    ):
        plan = (
            query_plan
            or QueryPlanner.build(query)
        )

        dense_results = self.dense.search(
            query=query,
            limit=retrieval_limit,
            software=software,
            software_version=software_version,
            project=project,
            content_type=content_type,
            language=language,
            human_language=human_language,
            query_plan=plan,
        )

        sparse_results = self.sparse.search(
            query=query,
            limit=retrieval_limit,
            software=software,
            software_version=software_version,
            project=project,
            content_type=content_type,
            language=language,
            human_language=human_language,
        )

        query_intent = plan.intent

        candidates = self._fuse_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
            candidate_limit=candidate_limit,
            query_intent=query_intent,
        )

        reranked = self.reranker.rerank(
            query=query,
            documents=candidates,
            limit=limit,
        )

        return self._expand_context(
            reranked,
            window=self.DEFAULT_CONTEXT_WINDOW,
        )

    def _expand_context(
        self,
        results,
        window=DEFAULT_CONTEXT_WINDOW,
    ):
        if not results or window <= 0:
            return results

        for item in results:
            payload = item.get("payload") or {}

            source = payload.get("source")
            file_hash = payload.get("file_hash")
            chunk_index = payload.get("chunk_index")
            anchor_text = payload.get("text") or ""

            if (
                not source
                or not file_hash
                or chunk_index is None
            ):
                item["context_window"] = 0
                item["context_chunks"] = [
                    {
                        "chunk_index": chunk_index,
                        "role": "anchor",
                        "text": anchor_text,
                    }
                ]
                item["expanded_context"] = anchor_text
                continue

            neighbors = self.dense.store.get_adjacent_chunks(
                source=source,
                file_hash=file_hash,
                chunk_index=chunk_index,
                window=window,
            )

            context_chunks = [
                {
                    "chunk_index": int(chunk_index),
                    "role": "anchor",
                    "text": anchor_text,
                }
            ]

            for neighbor in neighbors:
                neighbor_index = int(
                    neighbor["chunk_index"]
                )
                role = (
                    "before"
                    if neighbor_index < int(chunk_index)
                    else "after"
                )

                context_chunks.append(
                    {
                        "chunk_index": neighbor_index,
                        "role": role,
                        "text": (
                            neighbor.get("payload") or {}
                        ).get("text") or "",
                    }
                )

            context_chunks.sort(
                key=lambda chunk: chunk["chunk_index"]
            )

            item["context_window"] = int(window)
            item["context_chunks"] = context_chunks
            item["expanded_context"] = (
                "\n\n".join(
                    chunk["text"]
                    for chunk in context_chunks
                    if chunk["text"]
                )
            )

        return results

    def close(self):

        self.dense.close()
        self.sparse.close()
        self.reranker.close()
        VectorStore.shutdown()
