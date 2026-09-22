import re

from metadata.analyzers.base import (
    Symbol,
    SymbolAnalyzer,
)


class JavaScriptSymbolAnalyzer(SymbolAnalyzer):
    language = "javascript"

    PATTERNS = [
        (
            "class",
            re.compile(
                r"\bclass\s+"
                r"([A-Za-z_$][A-Za-z0-9_$]*)"
            ),
        ),
        (
            "function",
            re.compile(
                r"\bfunction\s+"
                r"([A-Za-z_$][A-Za-z0-9_$]*)"
                r"\s*\("
            ),
        ),
        (
            "arrow_function",
            re.compile(
                r"\b(?:const|let|var)\s+"
                r"([A-Za-z_$][A-Za-z0-9_$]*)"
                r"\s*=\s*(?:async\s*)?"
                r"\([^)]*\)\s*=>"
            ),
        ),
        (
            "method",
            re.compile(
                r"(?:^|[;{}])\s*"
                r"([A-Za-z_$][A-Za-z0-9_$]*)"
                r"\s*\([^)]*\)\s*\{",
                re.MULTILINE,
            ),
        ),
    ]

    def analyze(
        self,
        text: str,
    ) -> list[Symbol]:

        symbols = []

        for kind, pattern in self.PATTERNS:

            for match in pattern.finditer(text):

                name = match.group(1)

                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        language=self.language,
                        start=match.start(1),
                        end=match.end(),
                    )
                )

        symbols.sort(
            key=lambda symbol: (
                symbol.start,
                symbol.end,
            )
        )

        return self._attach_scopes(
            text,
            symbols,
        )

    @staticmethod
    def _attach_scopes(
        text: str,
        symbols: list[Symbol],
    ) -> list[Symbol]:

        for symbol in symbols:

            brace_start = text.find(
                "{",
                symbol.end,
            )

            if brace_start == -1:
                continue

            brace_end = (
                JavaScriptSymbolAnalyzer
                ._find_matching_brace(
                    text,
                    brace_start,
                )
            )

            if brace_end is None:
                continue

            symbol.scope_start = brace_start
            symbol.scope_end = brace_end

        return symbols

    @staticmethod
    def _find_matching_brace(
        text: str,
        opening_index: int,
    ) -> int | None:

        depth = 0
        in_line_comment = False
        in_block_comment = False
        in_string = False
        string_char = None
        escape = False

        i = opening_index

        while i < len(text):

            char = text[i]

            next_char = (
                text[i + 1]
                if i + 1 < len(text)
                else ""
            )

            if in_line_comment:

                if char == "\n":
                    in_line_comment = False

                i += 1
                continue

            if in_block_comment:

                if char == "*" and next_char == "/":
                    in_block_comment = False
                    i += 2
                    continue

                i += 1
                continue

            if in_string:

                if escape:
                    escape = False

                elif char == "\\":
                    escape = True

                elif char == string_char:
                    in_string = False
                    string_char = None

                i += 1
                continue

            if char == "/" and next_char == "/":
                in_line_comment = True
                i += 2
                continue

            if char == "/" and next_char == "*":
                in_block_comment = True
                i += 2
                continue

            if char in ('"', "'", "`"):
                in_string = True
                string_char = char
                i += 1
                continue

            if char == "{":
                depth += 1

            elif char == "}":

                depth -= 1

                if depth == 0:
                    return i

            i += 1

        return None