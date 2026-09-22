import re

from metadata.analyzers.base import (
    Symbol,
    SymbolAnalyzer,
)


class GenericSymbolAnalyzer(SymbolAnalyzer):
    """
    Conservative fallback analyzer.

    It intentionally does not assume a specific programming
    language. It detects common function-like declarations
    and uses brace scopes when available.
    """

    language = "generic"

    FUNCTION_PATTERN = re.compile(
        r"""
        (?:
            function\s+
            |
            def\s+
            |
            fn\s+
        )
        ([A-Za-z_][A-Za-z0-9_]*)
        \s*\(
        """,
        re.VERBOSE,
    )

    def analyze(
        self,
        text: str,
    ) -> list[Symbol]:

        symbols = []

        for match in self.FUNCTION_PATTERN.finditer(text):

            name = match.group(1)

            symbols.append(
                Symbol(
                    name=name,
                    kind="function",
                    language=self.language,
                    start=match.start(),
                    end=match.end(),
                )
            )

        return symbols