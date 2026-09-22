from pathlib import Path

from tree_sitter import Parser
from tree_sitter_language_pack import get_language


class TreeSitterParser:
    """
    Universal Tree-sitter parser wrapper.

    Responsible for:
        - loading a language grammar
        - creating a parser
        - parsing source text
    """

    def __init__(
        self,
        language: str,
    ):
        self.language_name = language

        language_object = get_language(
            language
        )

        self.parser = Parser(
            language_object
        )

    def parse(
        self,
        text: str,
    ):
        """
        Parse source text and return the
        Tree-sitter syntax tree.
        """

        if isinstance(text, str):
            text = text.encode(
                "utf-8"
            )

        return self.parser.parse(
            text
        )

    @staticmethod
    def language_from_path(
        path: Path,
    ) -> str | None:

        extension_map = {
            ".c": "c",
            ".h": "c",

            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".hpp": "cpp",
            ".hh": "cpp",
            ".hxx": "cpp",

            ".cs": "csharp",

            ".py": "python",

            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",

            ".ts": "typescript",
            ".tsx": "tsx",

            ".java": "java",

            ".go": "go",

            ".rs": "rust",

            ".lua": "lua",

            ".rb": "ruby",

            ".kt": "kotlin",
            ".kts": "kotlin",

            ".swift": "swift",

            ".php": "php",

            ".dart": "dart",

            ".hlsl": "hlsl",
            ".shader": "hlsl",

            ".glsl": "glsl",
            ".vert": "glsl",
            ".frag": "glsl",
        }

        return extension_map.get(
            path.suffix.lower()
        )