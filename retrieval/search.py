import re

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore


class SemanticSearch:
    """
    Dense semantic retrieval with structural and source-type awareness.

    Retrieval signals:
        1. Dense semantic similarity
        2. Explicit class/member structural matches
        3. Query-intent-aware source-type preference

    Source-type preference is intentionally kept small so metadata
    never overwhelms genuine semantic relevance.
    """

    # Small enough that semantic relevance remains the dominant signal.
    SOURCE_TYPE_WEIGHT = 0.08

    # Query-intent source preferences.
    #
    # These are relative priors, not hard filters.
    SOURCE_TYPE_PRIORS = {
        "configuration": {
            "rag_configuration": 1.00,
            "project_configuration": 0.85,
            "rag_code": 0.35,
            "project_code": 0.30,
            "rag_documentation": 0.25,
            "project_documentation": 0.20,
            "third_party_documentation": 0.10,
            "third_party_code": 0.05,
            "rag_test": -0.50,
            "project_test": -0.40,
        },
        "code": {
            "rag_code": 0.55,
            "project_code": 0.50,
            "rag_configuration": 0.30,
            "project_configuration": 0.20,
            "rag_documentation": 0.20,
            "project_documentation": 0.15,
            "third_party_code": 0.10,
            "third_party_documentation": 0.05,
            "rag_test": -0.10,
            "project_test": -0.10,
        },
        "documentation": {
            "rag_documentation": 0.55,
            "project_documentation": 0.50,
            "third_party_documentation": 0.35,
            "rag_code": 0.15,
            "project_code": 0.10,
            "rag_configuration": 0.10,
            "project_configuration": 0.05,
            "third_party_code": 0.05,
            "rag_test": -0.05,
            "project_test": -0.05,
        },
        "test": {
            "rag_test": 1.00,
            "project_test": 0.90,
            "rag_code": 0.20,
            "project_code": 0.15,
            "rag_configuration": 0.10,
            "project_configuration": 0.10,
            "rag_documentation": 0.10,
            "project_documentation": 0.10,
            "third_party_code": 0.05,
            "third_party_documentation": 0.05,
        },
        "general": {
            "rag_configuration": 0.20,
            "rag_code": 0.20,
            "rag_documentation": 0.15,
            "project_configuration": 0.15,
            "project_code": 0.15,
            "project_documentation": 0.15,
            "third_party_documentation": 0.05,
            "third_party_code": 0.05,
            "rag_test": -0.15,
            "project_test": -0.10,
        },
    }

    def __init__(self):
        self.embedder = Embedder()
        self.store = VectorStore()

    @staticmethod
    def _extract_identifiers(query):
        identifiers = re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            query,
        )

        return {
            identifier.lower()
            for identifier in identifiers
            if len(identifier) >= 3
        }

    @staticmethod
    def _extract_class_candidates(query):
        identifiers = re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            query,
        )

        return {
            identifier.lower()
            for identifier in identifiers
            if len(identifier) >= 3
            and identifier[0].isupper()
        }

    @staticmethod
    def _structural_score(
        payload,
        identifiers,
        class_candidates,
    ):
        score = 0.0

        class_name = (
            payload.get("class_name")
            or ""
        ).lower()

        member_name = (
            payload.get("member_name")
            or ""
        ).lower()

        chunk_type = (
            payload.get("chunk_type")
            or ""
        ).lower()

        # Exact class-name match.
        if (
            class_name
            and class_name in class_candidates
        ):
            score += 2.0

        # Exact member/method/property name match.
        if (
            member_name
            and member_name in identifiers
        ):
            score += 4.0

        # A member belonging to an explicitly requested class
        # gets an additional boost.
        if (
            class_name
            and class_name in class_candidates
            and member_name
        ):
            score += 1.0

        # Class declaration itself gets a smaller boost.
        if (
            chunk_type == "class"
            and class_name in class_candidates
        ):
            score += 0.5

        return score

    @staticmethod
    def _contains_exact_identifier(
        payload,
        identifiers,
    ):
        member_name = (
            payload.get("member_name")
            or ""
        ).lower()

        return (
            bool(member_name)
            and member_name in identifiers
        )

    @staticmethod
    def _normalize_query(query):
        if not query:
            return ""

        return re.sub(
            r"\s+",
            " ",
            query.strip().lower(),
        )

    @classmethod
    def _detect_query_intent(cls, query):
        """
        Detect the broad retrieval intent of a natural-language query.

        This intentionally uses simple transparent rules rather than
        another ML model. The result controls only a small source-type
        prior and never acts as a hard filter.
        """

        normalized = cls._normalize_query(query)

        if not normalized:
            return "general"

        test_patterns = (
            r"\btest\b",
            r"\btests\b",
            r"\btesting\b",
            r"\btest case\b",
            r"\btest cases\b",
            r"\bunit test\b",
            r"\bintegration test\b",
            r"\btest coverage\b",
            r"\bdiagnos",
            r"\bevaluation\b",
            r"\bevaluate\b",
            r"\bbenchmark\b",
            r"\bbenchmarks\b",
            r"\bregression test\b",
        )

        for pattern in test_patterns:
            if re.search(pattern, normalized):
                return "test"

        configuration_patterns = (
            r"\bconfig\b",
            r"\bconfiguration\b",
            r"\bconfigured\b",
            r"\bsetting\b",
            r"\bsettings\b",
            r"\boption\b",
            r"\boptions\b",
            r"\bparameter\b",
            r"\bparameters\b",
            r"\bvalue\b",
            r"\bdefault\b",
            r"\bdevice\b",
            r"\bmodel\b",
            r"\bmodels\b",
            r"\bpath\b",
            r"\bwhere is .* configured\b",
            r"\bwhere .* configured\b",
            r"\bwhat .* configured\b",
        )

        for pattern in configuration_patterns:
            if re.search(pattern, normalized):
                return "configuration"

        documentation_patterns = (
            r"\bdocumentation\b",
            r"\bdocument\b",
            r"\bdocs\b",
            r"\bguide\b",
            r"\bmanual\b",
            r"\bexplain\b",
            r"\bhow does\b",
            r"\bhow do\b",
            r"\bwhat does\b",
            r"\bwhy does\b",
        )

        for pattern in documentation_patterns:
            if re.search(pattern, normalized):
                return "documentation"

        code_patterns = (
            r"\bimplementation\b",
            r"\bimplemented\b",
            r"\bfunction\b",
            r"\bmethod\b",
            r"\bclass\b",
            r"\bproperty\b",
            r"\bfield\b",
            r"\bvariable\b",
            r"\bcode\b",
            r"\bsource code\b",
            r"\balgorithm\b",
            r"\blogic\b",
            r"\bcalled from\b",
            r"\bused by\b",
        )

        for pattern in code_patterns:
            if re.search(pattern, normalized):
                return "code"

        return "general"

    @classmethod
    def _source_type_score(
        cls,
        payload,
        query_intent,
    ):
        source_type = (
            payload.get("source_type")
            or ""
        ).strip().lower()

        if not source_type:
            return 0.0

        priors = cls.SOURCE_TYPE_PRIORS.get(
            query_intent,
            cls.SOURCE_TYPE_PRIORS["general"],
        )

        prior = priors.get(
            source_type,
            0.0,
        )

        return prior * cls.SOURCE_TYPE_WEIGHT

    def _get_structural_class_points(
        self,
        class_candidates,
        search_filter,
    ):
        if not class_candidates:
            return []

        points = []

        offset = None

        while True:
            batch, next_offset = (
                self.store.client.scroll(
                    collection_name=(
                        self.store.collection_name
                    ),
                    scroll_filter=search_filter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            )

            for point in batch:
                payload = point.payload or {}

                class_name = (
                    payload.get("class_name")
                    or ""
                ).lower()

                if class_name in class_candidates:
                    points.append(point)

            if next_offset is None:
                break

            offset = next_offset

        return points

    def search(
        self,
        query,
        limit=5,
        software=None,
        software_version=None,
        project=None,
        content_type=None,
        language=None,
        human_language=None,
    ):
        query_vector = self.embedder.encode(
            [query]
        )[0]

        search_filter = (
            self.store.build_filter(
                software=software,
                software_version=software_version,
                project=project,
                content_type=content_type,
                language=language,
                human_language=human_language,
            )
        )

        identifiers = (
            self._extract_identifiers(
                query
            )
        )

        class_candidates = (
            self._extract_class_candidates(
                query
            )
        )

        query_intent = self._detect_query_intent(
            query
        )

        retrieval_limit = max(
            limit * 10,
            50,
        )

        semantic_results = (
            self.store.client.query_points(
                collection_name=(
                    self.store.collection_name
                ),
                query=query_vector,
                query_filter=search_filter,
                limit=retrieval_limit,
                with_payload=True,
            )
        )

        points_by_id = {}

        semantic_scores = {}

        for point in semantic_results.points:
            points_by_id[point.id] = point
            semantic_scores[point.id] = (
                float(point.score)
            )

        # Retrieve every indexed chunk belonging to an explicitly
        # named class. This makes class-specific questions much
        # more reliable than semantic search alone.
        structural_points = (
            self._get_structural_class_points(
                class_candidates=class_candidates,
                search_filter=search_filter,
            )
        )

        for point in structural_points:
            points_by_id[point.id] = point

        ranked = []

        for point in points_by_id.values():
            payload = point.payload or {}

            semantic_score = semantic_scores.get(
                point.id,
                0.0,
            )

            structural_score = (
                self._structural_score(
                    payload=payload,
                    identifiers=identifiers,
                    class_candidates=class_candidates,
                )
            )

            source_type_score = (
                self._source_type_score(
                    payload=payload,
                    query_intent=query_intent,
                )
            )

            exact_member_match = (
                self._contains_exact_identifier(
                    payload,
                    identifiers,
                )
            )

            combined_score = (
                semantic_score
                + structural_score
                + source_type_score
            )

            ranked.append(
                {
                    "id": point.id,
                    "score": combined_score,
                    "semantic_score": semantic_score,
                    "structural_score": structural_score,
                    "source_type_score": source_type_score,
                    "query_intent": query_intent,
                    "exact_member_match": (
                        exact_member_match
                    ),
                    "payload": payload,
                }
            )

        # Exact member names are the strongest signal.
        # Then structural relevance.
        # Then combined semantic/source-type relevance.
        ranked.sort(
            key=lambda item: (
                item["exact_member_match"],
                item["structural_score"],
                item["score"],
            ),
            reverse=True,
        )

        return ranked[:limit]

    def close(self):
        self.store.close()