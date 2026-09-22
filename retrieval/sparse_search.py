import re

from rank_bm25 import BM25Okapi

from retrieval.lexical_index import LexicalIndex


class SparseSearch:
    """
    Code-aware BM25 sparse retrieval.

    BM25 provides lexical relevance while the structural score
    provides universal exact-identifier matching.

    The structural logic is intentionally generic. It does not
    contain project-specific knowledge or hard-coded symbol names.
    """

    def __init__(self):
        self.lexical_index = LexicalIndex()

        self.documents = list(
            self.lexical_index.documents.values()
        )

        self.bm25 = None

        self._build_bm25()

    def _build_bm25(self):
        if not self.documents:
            self.bm25 = None
            return

        corpus = [
            document.get(
                "tokens",
                [],
            )
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(
            corpus
        )

    @staticmethod
    def _extract_identifiers(
        query: str,
    ) -> set[str]:
        """
        Extract identifier-like query terms.

        Supports:
            BoneDriver
            LateUpdateBoneDriver
            find_bone
            findBone
            FindBone
        """

        if not query:
            return set()

        values = re.findall(
            r"[A-Za-z_][A-Za-z0-9_]*",
            query,
        )

        return {
            value.lower()
            for value in values
            if len(value) >= 2
        }

    @staticmethod
    def _identifier_components(
        value: str,
    ) -> set[str]:
        if not value:
            return set()

        return set(
            LexicalIndex.tokenize(value)
        )

    @classmethod
    def _structural_score(
        cls,
        query: str,
        payload: dict,
    ) -> float:
        """
        Universal exact-identifier scoring.

        Exact structural matches receive a stronger signal than
        ordinary BM25 component matches.

        This prevents:

            BoneDriverEditor

        from outranking:

            BoneDriver

        when the query explicitly asks for BoneDriver.
        """

        query_identifiers = cls._extract_identifiers(
            query
        )

        if not query_identifiers:
            return 0.0

        score = 0.0

        structural_values = [
            payload.get("symbol"),
            payload.get("class_name"),
            payload.get("member_name"),
        ]

        source = payload.get(
            "source",
            "",
        )

        path = payload.get(
            "path",
            "",
        )

        # --------------------------------------------------------------
        # Exact symbol/class/member matches
        # --------------------------------------------------------------

        for value in structural_values:
            if not value:
                continue

            normalized = str(value).lower()

            if normalized in query_identifiers:
                score += 8.0

        # --------------------------------------------------------------
        # Exact filename match
        # --------------------------------------------------------------

        for path_value in (
            source,
            path,
        ):
            if not path_value:
                continue

            filename = str(
                path_value
            ).replace(
                "\\",
                "/",
            ).rsplit(
                "/",
                1,
            )[-1]

            filename_stem = filename.rsplit(
                ".",
                1,
            )[0].lower()

            if filename_stem in query_identifiers:
                score += 7.0

        # --------------------------------------------------------------
        # Component overlap
        #
        # Useful for compound identifiers when an exact match is not
        # available.
        # --------------------------------------------------------------

        for value in structural_values:
            if not value:
                continue

            components = cls._identifier_components(
                str(value)
            )

            for query_identifier in query_identifiers:
                query_components = cls._identifier_components(
                    query_identifier
                )

                if not query_components:
                    continue

                overlap = (
                    components
                    & query_components
                )

                if overlap:
                    score += min(
                        len(overlap) * 0.5,
                        1.5,
                    )

        return score

    @staticmethod
    def _matches_filter(
        payload: dict,
        software: str | None = None,
        software_version: str | None = None,
        project: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
        human_language: str | None = None,
    ):
        filters = {
            "software": software,
            "software_version": software_version,
            "project": project,
            "content_type": content_type,
            "language": language,
            "human_language": human_language,
        }

        for field, value in filters.items():
            if (
                value is not None
                and payload.get(field) != value
            ):
                return False

        return True

    def search(
        self,
        query: str,
        limit: int = 5,
        software: str | None = None,
        software_version: str | None = None,
        project: str | None = None,
        content_type: str | None = None,
        language: str | None = None,
        human_language: str | None = None,
    ):
        if (
            not self.documents
            or self.bm25 is None
        ):
            return []

        query_tokens = (
            self.lexical_index.tokenize(
                query
            )
        )

        if not query_tokens:
            return []

        bm25_scores = self.bm25.get_scores(
            query_tokens
        )

        ranked = []

        for document, bm25_score in zip(
            self.documents,
            bm25_scores,
        ):
            payload = document.get(
                "payload",
                {},
            )

            if not self._matches_filter(
                payload=payload,
                software=software,
                software_version=software_version,
                project=project,
                content_type=content_type,
                language=language,
                human_language=human_language,
            ):
                continue

            structural_score = (
                self._structural_score(
                    query=query,
                    payload=payload,
                )
            )

            final_score = (
                float(bm25_score)
                + structural_score
            )

            ranked.append(
                {
                    "id": document["id"],
                    "score": final_score,
                    "bm25_score": float(
                        bm25_score
                    ),
                    "structural_score": structural_score,
                    "payload": payload,
                }
            )

        ranked.sort(
            key=lambda item: (
                item["score"],
                item["structural_score"],
                item["bm25_score"],
            ),
            reverse=True,
        )

        return ranked[:limit]

    def close(self):
        self.documents = []
        self.bm25 = None