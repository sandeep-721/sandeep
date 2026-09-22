import re

from metadata.analyzers.base import (
    Symbol,
    SymbolAnalyzer,
)


class GoSymbolAnalyzer(SymbolAnalyzer):
    """
    Go source-code symbol analyzer.

    Detects:
        - structs
        - interfaces
        - functions
        - methods
    """

    language = "go"

    TYPE_PATTERN = re.compile(
        r"\btype\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)"
        r"\s+(struct|interface)\b"
    )

    FUNCTION_PATTERN = re.compile(
        r"\bfunc\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)"
        r"\s*\("
    )

    METHOD_PATTERN = re.compile(
        r"\bfunc\s*"
        r"\([^)]*\)\s*"
        r"([A-Za-z_][A-Za-z0-9_]*)"
        r"\s*\("
    )

    def analyze(
        self,
        text: str,
    ) -> list[Symbol]:

        symbols = []

        symbols.extend(
            self._detect_types(text)
        )

        symbols.extend(
            self._detect_functions(text)
        )

        symbols.extend(
            self._detect_methods(text)
        )

        symbols.sort(
            key=lambda symbol: (
                symbol.start,
                symbol.end,
            )
        )

        self._attach_scopes(
            text,
            symbols,
        )

        return symbols

    def _detect_types(
        self,
        text: str,
    ) -> list[Symbol]:

        symbols = []

        for match in self.TYPE_PATTERN.finditer(text):

            name = match.group(1)
            kind = match.group(2)

            symbols.append(
                Symbol(
                    name=name,
                    kind=kind,
                    language=self.language,
                    start=match.start(1),
                    end=match.end(1),
                )
            )

        return symbols

    def _detect_functions(
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
                    start=match.start(1),
                    end=match.end(),
                )
            )

        return symbols

    def _detect_methods(
        self,
        text: str,
    ) -> list[Symbol]:

        symbols = []

        for match in self.METHOD_PATTERN.finditer(text):

            name = match.group(1)

            symbols.append(
                Symbol(
                    name=name,
                    kind="method",
                    language=self.language,
                    start=match.start(1),
                    end=match.end(),
                )
            )

        return symbols

    @staticmethod
    def _attach_scopes(
        text: str,
        symbols: list[Symbol],
    ) -> None:

        for symbol in symbols:

            brace_start = text.find(
                "{",
                symbol.end,
            )

            if brace_start == -1:
                continue

            brace_end = (
                GoSymbolAnalyzer._find_matching_brace(
                    text,
                    brace_start,
                )
            )

            if brace_end is None:
                continue

            symbol.scope_start = brace_start
            symbol.scope_end = brace_end

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