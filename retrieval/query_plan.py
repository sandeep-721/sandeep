from __future__ import annotations

from dataclasses import dataclass
import re

from retrieval.query_intent import QueryIntentDetector


@dataclass(frozen=True)
class QueryPlan:
    query: str
    intent: str
    identifiers: frozenset[str]
    class_candidates: frozenset[str]
    explicit_identifier_query: bool

    @property
    def has_identifiers(self) -> bool:
        return bool(self.identifiers)

    @property
    def has_class_candidates(self) -> bool:
        return bool(self.class_candidates)

    @property
    def retrieval_modes(self) -> tuple[str, ...]:
        modes = ["dense", "sparse"]

        if self.explicit_identifier_query or self.intent == "code":
            modes.append("structural")

        return tuple(modes)

    @property
    def retrieval_weights(self) -> tuple[float, float]:
        return QueryPlanner.RETRIEVAL_WEIGHTS.get(
            self.intent,
            QueryPlanner.RETRIEVAL_WEIGHTS["general"],
        )

    @property
    def use_structural_signals(self) -> bool:
        return (
            self.explicit_identifier_query
            or self.intent == "code"
        )


class QueryPlanner:
    RETRIEVAL_WEIGHTS = {
        "code": (1.25, 0.75),
        "test": (1.25, 0.75),
        "documentation": (0.90, 1.10),
        "configuration": (0.80, 1.20),
        "history": (0.85, 1.15),
        "general": (1.00, 1.00),
    }

    """Build one deterministic query plan shared by retrieval stages."""

    @staticmethod
    def _extract_identifiers(query: str) -> frozenset[str]:
        identifiers = re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            query or "",
        )

        return frozenset(
            identifier.lower()
            for identifier in identifiers
            if len(identifier) >= 3
        )

    @staticmethod
    def _extract_class_candidates(query: str) -> frozenset[str]:
        identifiers = re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            query or "",
        )

        return frozenset(
            identifier.lower()
            for identifier in identifiers
            if len(identifier) >= 3
            and identifier[0].isupper()
        )

    @staticmethod
    def _is_explicit_identifier_query(query: str) -> bool:
        normalized = (query or "").strip()

        if not normalized:
            return False

        normalized = normalized.strip("\"'`.,:;()[]{}")

        if not normalized:
            return False

        if re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_.-]*",
            normalized,
        ):
            return True

        tokens = normalized.split()

        if not tokens or len(tokens) > 3:
            return False

        return all(
            re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_.-]*",
                token.strip("\"'`.,:;()[]{}"),
            )
            for token in tokens
        )

    @classmethod
    def build(cls, query: str) -> QueryPlan:
        return QueryPlan(
            query=query or "",
            intent=QueryIntentDetector.detect(query),
            identifiers=cls._extract_identifiers(query),
            class_candidates=cls._extract_class_candidates(query),
            explicit_identifier_query=cls._is_explicit_identifier_query(query),
        )


build_query_plan = QueryPlanner.build
