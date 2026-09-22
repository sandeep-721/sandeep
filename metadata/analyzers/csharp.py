import re

from metadata.analyzers.base import (
    Symbol,
    SymbolAnalyzer,
)


class CSharpSymbolAnalyzer(SymbolAnalyzer):
    language = "csharp"

    PATTERNS = [
        (
            "enum",
            re.compile(
                r"\b(?:public|private|protected|internal|static|sealed|partial|\s)*"
                r"\benum\s+([A-Za-z_][A-Za-z0-9_]*)"
            ),
        ),
        (
            "interface",
            re.compile(
                r"\b(?:public|private|protected|internal|static|partial|\s)*"
                r"\binterface\s+([A-Za-z_][A-Za-z0-9_]*)"
            ),
        ),
        (
            "struct",
            re.compile(
                r"\b(?:public|private|protected|internal|static|readonly|partial|\s)*"
                r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)"
            ),
        ),
        (
            "class",
            re.compile(
                r"\b(?:public|private|protected|internal|abstract|sealed|static|partial|\s)*"
                r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)"
            ),
        ),
        (
            "method",
            re.compile(
                r"\b(?:public|private|protected|internal|static|virtual|override|"
                r"abstract|async|sealed|extern|new|unsafe|partial|\s)+"
                r"(?:[A-Za-z_][A-Za-z0-9_<>,\[\]?\.]*\s+)"
                r"([A-Za-z_][A-Za-z0-9_]*)\s*\("
            ),
        ),
    ]

    def analyze(self, text: str) -> list[Symbol]:
        symbols = []

        for kind, pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1)

                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        language=self.language,
                        start=match.start(),
                        end=match.end(),
                    )
                )

        symbols.sort(
            key=lambda symbol: (
                symbol.start,
                symbol.end,
            )
        )

        return self._attach_scopes(text, symbols)

    def _attach_scopes(
        self,
        text: str,
        symbols: list[Symbol],
    ) -> list[Symbol]:

        result = []

        for symbol in symbols:
            brace_start = self._find_open_brace(
                text,
                symbol.end,
            )

            if brace_start is None:
                result.append(symbol)
                continue

            brace_end = self._find_matching_brace(
                text,
                brace_start,
            )

            if brace_end is None:
                result.append(symbol)
                continue

            symbol.scope_start = brace_start
            symbol.scope_end = brace_end

            result.append(symbol)

        return result

    @staticmethod
    def _find_open_brace(
        text: str,
        start: int,
    ) -> int | None:

        index = text.find("{", start)

        if index == -1:
            return None

        return index

    @staticmethod
    def _find_matching_brace(
        text: str,
        opening_index: int,
    ) -> int | None:

        depth = 0
        in_line_comment = False
        in_block_comment = False
        in_string = False
        in_char = False
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
                elif char == '"':
                    in_string = False

                i += 1
                continue

            if in_char:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == "'":
                    in_char = False

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

            if char == '"':
                in_string = True
                i += 1
                continue

            if char == "'":
                in_char = True
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