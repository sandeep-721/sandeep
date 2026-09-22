from pathlib import Path

from metadata.analyzers.base import (
    Symbol,
    SymbolAnalyzer,
    build_symbol_hierarchy,
)

from metadata.analyzers.c import (
    CSymbolAnalyzer,
)

from metadata.analyzers.cpp import (
    CppSymbolAnalyzer,
)

from metadata.analyzers.csharp import (
    CSharpSymbolAnalyzer,
)

from metadata.analyzers.go import (
    GoSymbolAnalyzer,
)

from metadata.analyzers.java import (
    JavaSymbolAnalyzer,
)

from metadata.analyzers.javascript import (
    JavaScriptSymbolAnalyzer,
)

from metadata.analyzers.python import (
    PythonSymbolAnalyzer,
)

from metadata.tree_sitter.symbols import (
    TreeSitterSymbolExtractor,
)


EXTENSION_LANGUAGE_MAP = {
    # C#
    ".cs": "csharp",

    # Python
    ".py": "python",

    # JavaScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",

    # TypeScript
    ".ts": "typescript",
    ".tsx": "tsx",

    # C
    ".c": "c",
    ".h": "c",

    # C++
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".hxx": "cpp",

    # Java
    ".java": "java",

    # Go
    ".go": "go",

    # Rust
    ".rs": "rust",

    # Lua
    ".lua": "lua",

    # Ruby
    ".rb": "ruby",

    # Kotlin
    ".kt": "kotlin",
    ".kts": "kotlin",

    # Swift
    ".swift": "swift",

    # PHP
    ".php": "php",

    # Dart
    ".dart": "dart",

    # HLSL
    ".hlsl": "hlsl",
    ".shader": "hlsl",

    # GLSL
    ".glsl": "glsl",
    ".vert": "glsl",
    ".frag": "glsl",
}


class UniversalSymbolAnalyzer:
    """
    Universal entry point for source-code symbol analysis.

    Tree-sitter is the primary parser.

    Existing language-specific analyzers are retained as
    compatibility fallbacks when Tree-sitter cannot parse
    a language or does not produce usable symbols.

    The rest of Universal RAG only interacts with this class.
    """

    def __init__(self):
        self.analyzers: list[SymbolAnalyzer] = [
            CSharpSymbolAnalyzer(),
            PythonSymbolAnalyzer(),
            JavaScriptSymbolAnalyzer(),
            CSymbolAnalyzer(),
            CppSymbolAnalyzer(),
            JavaSymbolAnalyzer(),
            GoSymbolAnalyzer(),
        ]

        self._by_language = {
            analyzer.language: analyzer
            for analyzer in self.analyzers
        }

    def detect_language(
        self,
        path: Path,
    ) -> str:
        """
        Convert a file extension into the canonical
        language identifier used by Universal RAG.
        """

        extension = path.suffix.lower()

        return EXTENSION_LANGUAGE_MAP.get(
            extension,
            "generic",
        )

    def get_analyzer(
        self,
        language: str,
    ) -> SymbolAnalyzer:
        """
        Return the legacy/fallback analyzer for a language.

        Tree-sitter is handled separately by analyze().
        """

        analyzer = self._by_language.get(
            language
        )

        if analyzer is not None:
            return analyzer

        from metadata.analyzers.generic import (
            GenericSymbolAnalyzer,
        )

        return GenericSymbolAnalyzer()

    @staticmethod
    def _tree_sitter_available(
        language: str,
    ) -> bool:
        """
        Check whether Tree-sitter has a grammar for the language.

        The check is intentionally isolated so a missing grammar
        never breaks the entire indexing pipeline.
        """

        try:
            from tree_sitter_language_pack import (
                get_language,
            )

            get_language(language)

            return True

        except Exception:
            return False

    def _analyze_with_tree_sitter(
        self,
        text: str,
        language: str,
    ) -> list[Symbol]:
        """
        Analyze source code using Tree-sitter.

        Returns an empty list when Tree-sitter cannot handle
        the requested language.
        """

        if language == "generic":
            return []

        if not self._tree_sitter_available(
            language
        ):
            return []

        try:
            extractor = TreeSitterSymbolExtractor(
                language
            )

            symbols = extractor.extract(
                text
            )

            if not symbols:
                return []

            return symbols

        except Exception:
            return []

    def _analyze_with_fallback(
        self,
        text: str,
        language: str,
    ) -> list[Symbol]:
        """
        Analyze source using the existing language-specific
        analyzer.

        This preserves compatibility with languages or syntax
        that the Tree-sitter extractor does not yet understand.
        """

        analyzer = self.get_analyzer(
            language
        )

        try:
            return analyzer.analyze(
                text
            )

        except Exception:
            return []

    @staticmethod
    def _has_usable_symbols(
        symbols: list[Symbol],
    ) -> bool:
        """
        Determine whether symbol extraction produced meaningful
        symbols rather than malformed or empty results.
        """

        if not symbols:
            return False

        for symbol in symbols:
            if not symbol.name:
                continue

            if not symbol.kind:
                continue

            if symbol.start < 0:
                continue

            if symbol.end < symbol.start:
                continue

            return True

        return False

    def analyze(
        self,
        text: str,
        language: str,
    ) -> list[Symbol]:
        """
        Perform universal symbol analysis.

        Priority:

            1. Tree-sitter
            2. Existing language analyzer fallback
            3. Empty result

        Parent/child relationships are always constructed
        through the language-independent hierarchy builder.
        """

        symbols = self._analyze_with_tree_sitter(
            text=text,
            language=language,
        )

        if not self._has_usable_symbols(
            symbols
        ):
            symbols = self._analyze_with_fallback(
                text=text,
                language=language,
            )

        if not symbols:
            return []

        return build_symbol_hierarchy(
            symbols
        )

    def analyze_file(
        self,
        path: Path,
    ) -> list[Symbol]:
        """
        Analyze a source file using the universal pipeline.
        """

        language = self.detect_language(
            path
        )

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return self.analyze(
            text=text,
            language=language,
        )


def detect_symbols(
    text: str,
    extension: str,
) -> list[Symbol]:
    """
    Backward-compatible universal symbol detection.

    Existing callers can continue passing:

        detect_symbols(text, ".cs")

    without knowing whether Tree-sitter or a fallback
    analyzer is being used internally.
    """

    analyzer = UniversalSymbolAnalyzer()

    language = analyzer.detect_language(
        Path(
            f"source{extension}"
        )
    )

    return analyzer.analyze(
        text=text,
        language=language,
    )


def detect_primary_symbol(
    symbols: list[Symbol],
) -> Symbol | None:
    """
    Return the earliest symbol in the source.
    """

    if not symbols:
        return None

    return min(
        symbols,
        key=lambda symbol: (
            symbol.start,
            symbol.end,
        ),
    )


def find_symbols_containing_offset(
    symbols: list[Symbol],
    offset: int,
) -> list[Symbol]:
    """
    Return all symbols whose scope contains the supplied
    character/byte offset.
    """

    return [
        symbol
        for symbol in symbols
        if symbol.contains(offset)
    ]


def find_symbol_at_offset(
    symbols: list[Symbol],
    offset: int,
) -> Symbol | None:
    """
    Return the most specific symbol containing the offset.
    """

    candidates = find_symbols_containing_offset(
        symbols,
        offset,
    )

    if not candidates:
        return None

    return min(
        candidates,
        key=lambda symbol: (
            symbol.scope_end - symbol.scope_start
            if symbol.scope_start is not None
            and symbol.scope_end is not None
            else float("inf")
        ),
    )