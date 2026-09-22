from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DocumentMetadata:
    source: str
    content_type: str

    software: Optional[str] = None
    software_version: Optional[str] = None

    project: Optional[str] = None
    repository: Optional[str] = None
    branch: Optional[str] = None

    # Programming / document format language.
    # Examples: csharp, python, javascript, markdown, json.
    language: Optional[str] = None

    # Natural/human language detected from the content.
    # Examples: en, te, ta, hi.
    human_language: Optional[str] = None

    path: Optional[str] = None

    symbol: Optional[str] = None
    symbol_kind: Optional[str] = None

    file_hash: Optional[str] = None

    chunk_index: Optional[int] = None
    chunk_start: Optional[int] = None
    chunk_end: Optional[int] = None

    source_type: Optional[str] = None
    source_version: Optional[str] = None

    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if value is not None
        }