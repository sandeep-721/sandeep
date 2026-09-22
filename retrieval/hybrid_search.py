from retrieval.search import SemanticSearch
from retrieval.vector_store import VectorStore
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

    def __init__(self):
        self.dense = SemanticSearch()
        self.sparse = SparseSearch()
        self.reranker = Reranker()

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

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

        rrf_score = cls._rrf_score(rank)

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
    ):
        """
        Standard chunk-level RRF.

        Dense and sparse retrieval are fused directly.
        No source-level aggregation or source-diversity
        manipulation is performed.
        """

        fused = {}

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            cls._add_result(
                fused=fused,
                result=result,
                rank=rank,
                retriever="dense",
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
    ):
        dense_results = self.dense.search(
            query=query,
            limit=retrieval_limit,
            software=software,
            software_version=software_version,
            project=project,
            content_type=content_type,
            language=language,
            human_language=human_language,
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

        candidates = self._fuse_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
            candidate_limit=candidate_limit,
        )

        reranked = self.reranker.rerank(
            query=query,
            documents=candidates,
            limit=limit,
        )

        return reranked

    def close(self):
        self.dense.close()
        self.sparse.close()
        self.reranker.close()
        VectorStore.shutdown()
