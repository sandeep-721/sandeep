from dataclasses import asdict, dataclass
import re
from typing import Dict, Tuple


_CITATION_PATTERN = re.compile(
    r"\[((?:E\d+)(?:\s*[,;]\s*E\d+)*)\]"
)

_SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?!\[(?:E\d+)(?:\s*[,;]\s*E\d+)*\])|\n+"
)

_TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
)

_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by",
        "does", "for", "from", "has", "have", "how", "in",
        "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "uses", "use", "using", "what",
        "which", "with",
    }
)


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    citations: Tuple[str, ...]
    substantive: bool

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CitationSupport:
    claim_id: str
    evidence_id: str
    score: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    claims: Tuple[Claim, ...]
    supports: Tuple[CitationSupport, ...]
    invalid_citations: Tuple[str, ...]
    uncited_claims: Tuple[str, ...]
    weak_support: Tuple[str, ...]
    citation_coverage: float
    verified: bool

    def to_dict(self):
        return {
            "claims": [
                claim.to_dict()
                for claim in self.claims
            ],
            "supports": [
                support.to_dict()
                for support in self.supports
            ],
            "invalid_citations": list(
                self.invalid_citations
            ),
            "uncited_claims": list(
                self.uncited_claims
            ),
            "weak_support": list(
                self.weak_support
            ),
            "citation_coverage": self.citation_coverage,
            "verified": self.verified,
        }


class ClaimVerifier:
    MIN_CLAIM_WORDS = 4
    MIN_SUPPORT_SCORE = 0.12

    @staticmethod
    def _tokens(text):
        tokens = {
            token.lower()
            for token in _TOKEN_PATTERN.findall(text or "")
        }

        return {
            token
            for token in tokens
            if token not in _STOP_WORDS
        }

    @classmethod
    def _extract_citations(cls, text):
        citations = []

        for match in _CITATION_PATTERN.finditer(text or ""):
            for citation in re.split(
                r"[,;]\s*",
                match.group(1),
            ):
                citation = citation.strip()

                if citation and citation not in citations:
                    citations.append(citation)

        return tuple(citations)

    @classmethod
    def _is_substantive(cls, text):
        cleaned = re.sub(
            r"\[(?:E\d+)(?:\s*[,;]\s*E\d+)*\]",
            "",
            text or "",
        ).strip()

        if not cleaned:
            return False

        if cleaned == (
            "The indexed project evidence does not confirm this."
        ):
            return False

        words = re.findall(
            r"\b[A-Za-z0-9_]+\b",
            cleaned,
        )

        return len(words) >= cls.MIN_CLAIM_WORDS

    @classmethod
    def extract_claims(cls, answer):
        claims = []

        normalized = re.sub(
            r"(?<=[.!?])\s+(?=\[(?:E\d+))",
            " ",
            (answer or "").strip(),
        )

        for index, raw in enumerate(
            _SENTENCE_PATTERN.split(normalized),
            start=1,
        ):
            text = raw.strip()

            if not text:
                continue

            claims.append(
                Claim(
                    claim_id=f"C{index}",
                    text=text,
                    citations=cls._extract_citations(
                        text
                    ),
                    substantive=cls._is_substantive(
                        text
                    ),
                )
            )

        return tuple(claims)

    @classmethod
    def _evidence_tokens(cls, evidence_packet):
        return {
            item.evidence_id: cls._tokens(
                item.expanded_context or item.text
            )
            for item in evidence_packet.evidence
        }

    @classmethod
    def _support_score(
        cls,
        claim_text,
        evidence_tokens,
    ):
        claim_tokens = cls._tokens(claim_text)

        if not claim_tokens or not evidence_tokens:
            return 0.0

        overlap = claim_tokens & evidence_tokens

        return len(overlap) / len(claim_tokens)

    def verify(self, answer, evidence_packet):
        claims = self.extract_claims(answer)

        valid_evidence_ids = {
            item.evidence_id
            for item in evidence_packet.evidence
        }

        evidence_tokens = self._evidence_tokens(
            evidence_packet
        )

        invalid_citations = []
        uncited_claims = []
        weak_support = []
        supports = []

        substantive_claims = [
            claim
            for claim in claims
            if claim.substantive
        ]

        cited_claims = 0

        for claim in substantive_claims:
            valid_claim_citations = []

            for citation in claim.citations:
                if citation not in valid_evidence_ids:
                    if citation not in invalid_citations:
                        invalid_citations.append(
                            citation
                        )
                    continue

                valid_claim_citations.append(
                    citation
                )

                score = self._support_score(
                    claim.text,
                    evidence_tokens[citation],
                )

                supports.append(
                    CitationSupport(
                        claim_id=claim.claim_id,
                        evidence_id=citation,
                        score=score,
                    )
                )

            if not valid_claim_citations:
                uncited_claims.append(
                    claim.claim_id
                )
                continue

            cited_claims += 1

            best_score = max(
                (
                    support.score
                    for support in supports
                    if support.claim_id == claim.claim_id
                ),
                default=0.0,
            )

            if best_score < self.MIN_SUPPORT_SCORE:
                weak_support.append(
                    claim.claim_id
                )

        citation_coverage = (
            cited_claims / len(substantive_claims)
            if substantive_claims
            else 1.0
        )

        verified = (
            not invalid_citations
            and not uncited_claims
            and not weak_support
        )

        return VerificationResult(
            claims=claims,
            supports=tuple(supports),
            invalid_citations=tuple(
                invalid_citations
            ),
            uncited_claims=tuple(
                uncited_claims
            ),
            weak_support=tuple(
                weak_support
            ),
            citation_coverage=citation_coverage,
            verified=verified,
        )
