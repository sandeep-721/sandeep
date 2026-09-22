from dataclasses import dataclass
from typing import Optional


@dataclass
class Symbol:
    """
    Universal language-independent representation of a code symbol.
    """

    name: str
    kind: str
    language: str

    start: int
    end: int

    scope_start: Optional[int] = None
    scope_end: Optional[int] = None

    parent_symbol: Optional[str] = None
    parent_kind: Optional[str] = None

    @property
    def has_scope(self) -> bool:
        return (
            self.scope_start is not None
            and self.scope_end is not None
        )

    def contains(self, offset: int) -> bool:
        if not self.has_scope:
            return False

        return (
            self.scope_start <= offset <= self.scope_end
        )


class SymbolAnalyzer:
    """
    Base interface for language-specific analyzers.
    """

    language = "unknown"

    def analyze(
        self,
        text: str,
    ) -> list[Symbol]:
        raise NotImplementedError

    def supports(
        self,
        language: str | None,
    ) -> bool:
        return language == self.language


def normalize_symbol_kind(kind):
    """
    Normalize parser-specific symbol names while
    preserving the difference between functions
    and methods.
    """

    aliases = {
        "async_function": "function",
        "arrow_function": "function",
    }

    return aliases.get(kind, kind)


def normalize_symbols(
    symbols: list[Symbol],
) -> list[Symbol]:
    """
    Normalize symbol kinds without changing the
    language-specific analyzers.
    """

    for symbol in symbols:
        symbol.kind = normalize_symbol_kind(
            symbol.kind
        )

    return symbols


def build_symbol_hierarchy(
    symbols: list[Symbol],
) -> list[Symbol]:
    """
    Build parent/child relationships using symbol scopes.

    The hierarchy logic is language-independent.
    """

    normalize_symbols(symbols)

    for symbol in symbols:
        symbol.parent_symbol = None
        symbol.parent_kind = None

    for symbol in symbols:

        if not symbol.has_scope:
            continue

        candidates = []

        for candidate in symbols:

            if candidate is symbol:
                continue

            if not candidate.has_scope:
                continue

            if not (
                candidate.scope_start <= symbol.start
                and candidate.scope_end >= symbol.end
            ):
                continue

            candidates.append(candidate)

        if not candidates:
            continue

        parent = min(
            candidates,
            key=lambda item: (
                item.scope_end - item.scope_start
            ),
        )

        symbol.parent_symbol = parent.name
        symbol.parent_kind = parent.kind

    return symbols