import ast

from metadata.analyzers.base import (
    Symbol,
    SymbolAnalyzer,
)


class PythonSymbolAnalyzer(SymbolAnalyzer):
    language = "python"

    def analyze(
        self,
        text: str,
    ) -> list[Symbol]:

        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []

        symbols = []

        for node in ast.walk(tree):

            if isinstance(
                node,
                (
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                if isinstance(node, ast.ClassDef):
                    kind = "class"

                elif isinstance(
                    node,
                    ast.AsyncFunctionDef,
                ):
                    kind = "async_function"

                else:
                    kind = "function"

                start = self._offset(
                    text,
                    node.lineno,
                    node.col_offset,
                )

                end = self._end_offset(
                    text,
                    node,
                )

                symbols.append(
                    Symbol(
                        name=node.name,
                        kind=kind,
                        language=self.language,
                        start=start,
                        end=end,
                        scope_start=start,
                        scope_end=end,
                    )
                )

        symbols.sort(
            key=lambda symbol: (
                symbol.start,
                symbol.end,
            )
        )

        return symbols

    @staticmethod
    def _offset(
        text: str,
        line: int,
        column: int,
    ) -> int:

        lines = text.splitlines(
            keepends=True
        )

        return (
            sum(
                len(value)
                for value in lines[:line - 1]
            )
            + column
        )

    @staticmethod
    def _end_offset(
        text: str,
        node,
    ) -> int:

        if not hasattr(
            node,
            "end_lineno",
        ):
            return (
                PythonSymbolAnalyzer._offset(
                    text,
                    node.lineno,
                    node.col_offset,
                )
            )

        return PythonSymbolAnalyzer._offset(
            text,
            node.end_lineno,
            node.end_col_offset,
        )