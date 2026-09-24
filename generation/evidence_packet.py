from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    rank: int
    source: str
    file_hash: Optional[str]
    chunk_index: Optional[int]
    chunk_start: Optional[int]
    chunk_end: Optional[int]
    text: str
    expanded_context: str
    context_window: int
    context_chunks: Tuple[Dict[str, Any], ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)

    def render(self):
        lines = [
            f"[{self.evidence_id}]",
            f"File: {self.source}",
        ]

        if self.chunk_index is not None:
            lines.append(f"Chunk: {self.chunk_index}")

        if self.chunk_start is not None:
            lines.append(
                f"Range: {self.chunk_start}-{self.chunk_end}"
            )

        for key, value in self.metadata.items():
            if value is None or value == "":
                continue

            label = key.replace("_", " ").title()
            lines.append(f"{label}: {value}")

        lines.extend(
            [
                "Evidence:",
                self.expanded_context or self.text,
            ]
        )

        return "\n".join(lines)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSource:
    source: str
    evidence_ids: Tuple[str, ...]
    chunks: Tuple[int, ...]
    project: Optional[str]
    software: Optional[str]
    software_version: Optional[str]
    language: Optional[str]

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EvidencePacket:
    query: str
    evidence: Tuple[EvidenceItem, ...]
    sources: Tuple[EvidenceSource, ...]
    max_chars: int

    @property
    def text(self):
        return "\n\n".join(
            item.render()
            for item in self.evidence
        )

    def to_dict(self):
        return {
            "query": self.query,
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "sources": [
                source.to_dict()
                for source in self.sources
            ],
            "max_chars": self.max_chars,
        }

    def is_empty(self):
        return not self.evidence
