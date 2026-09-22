from metadata.analyzers.base import Symbol
from metadata.tree_sitter.parser import TreeSitterParser


class TreeSitterSymbolExtractor:
    """
    Converts Tree-sitter syntax trees into the
    Universal RAG Symbol representation.
    """

    NODE_KIND_MAP = {
        # C#
        "class_declaration": "class",
        "struct_declaration": "struct",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "method_declaration": "method",
        "constructor_declaration": "method",

        # Python
        "class_definition": "class",
        "function_definition": "function",

        # JavaScript / TypeScript
        "function_declaration": "function",
        "method_definition": "method",
        "arrow_function": "function",

        # C / C++
        "struct_specifier": "struct",
        "class_specifier": "class",
        "function_definition": "function",

        # Java
        "interface_declaration": "interface",
        "enum_declaration": "enum",

        # Go
        "type_declaration": "type",
        "function_declaration": "function",
        "method_declaration": "method",

        # Rust
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "impl_item": "impl",
        "function_item": "function",

        # HLSL / GLSL
        "function_definition": "function",

        # Ruby
        "method": "method",

        # PHP
        "method_declaration": "method",
        "function_definition": "function",

        # Kotlin
        "function_declaration": "function",

        # Swift
        "function_declaration": "function",

        # Dart
        "method_signature": "method",
        "function_signature": "function",
    }

    def __init__(self, language: str):
        self.language = language
        self.parser = TreeSitterParser(language)

    def extract(self, text: str) -> list[Symbol]:
        tree = self.parser.parse(text)

        symbols = []

        self._walk(
            tree.root_node,
            text,
            symbols,
        )

        return symbols

    def _walk(
        self,
        node,
        text: str,
        symbols: list[Symbol],
    ):
        # Dart method_signature contains a nested
        # function_signature. The outer node is the
        # actual class method, so do not emit the
        # nested function_signature separately.
        if (
            self.language == "dart"
            and node.type == "function_signature"
            and node.parent is not None
            and node.parent.type == "method_signature"
        ):
            return

        kind = self._get_kind(node)

        if kind is not None:
            name = self._extract_name(
                node,
                text,
            )

            if name:
                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        language=self.language,
                        start=node.start_byte,
                        end=node.end_byte,
                        scope_start=node.start_byte,
                        scope_end=node.end_byte,
                    )
                )

        for child in node.children:
            self._walk(
                child,
                text,
                symbols,
            )

    def _get_kind(
        self,
        node,
    ) -> str | None:

        node_type = node.type

        # Rust:
        # function_item inside impl_item = method.
        # Top-level function_item = function.
        if (
            self.language == "rust"
            and node_type == "function_item"
        ):
            if self._has_parent_type(
                node,
                {"impl_item"},
            ):
                return "method"

            return "function"

        # Kotlin:
        # function_declaration inside a class/object
        # = method. Top-level = function.
        if (
            self.language == "kotlin"
            and node_type == "function_declaration"
        ):
            if self._has_parent_type(
                node,
                {
                    "class_declaration",
                    "object_declaration",
                    "companion_object",
                },
            ):
                return "method"

            return "function"

        # Swift:
        # function_declaration inside a type
        # = method. Top-level = function.
        if (
            self.language == "swift"
            and node_type == "function_declaration"
        ):
            if self._has_parent_type(
                node,
                {
                    "class_declaration",
                    "struct_declaration",
                    "enum_declaration",
                    "extension_declaration",
                },
            ):
                return "method"

            return "function"

        # Python.
        if (
            self.language == "python"
            and node_type == "function_definition"
        ):
            return "function"

        # Go.
        if (
            self.language == "go"
            and node_type == "function_declaration"
        ):
            return "function"

        if (
            self.language == "go"
            and node_type == "method_declaration"
        ):
            return "method"

        return self.NODE_KIND_MAP.get(node_type)

    @staticmethod
    def _has_parent_type(
        node,
        parent_types: set[str],
    ) -> bool:
        """
        Check the complete ancestor chain for a semantic
        parent type.
        """

        parent = node.parent

        while parent is not None:
            if parent.type in parent_types:
                return True

            parent = parent.parent

        return False

    def _extract_name(
        self,
        node,
        text: str,
    ) -> str | None:

        # Most Tree-sitter grammars expose declaration
        # names through the "name" field.
        name_node = node.child_by_field_name("name")

        if name_node is not None:
            value = self._node_text(
                name_node,
                text,
            )

            if value:
                return value

        # Dart method_signature contains a nested
        # function_signature containing the identifier.
        if (
            self.language == "dart"
            and node.type == "method_signature"
        ):
            for child in node.children:
                if child.type == "function_signature":
                    value = self._extract_name(
                        child,
                        text,
                    )

                    if value:
                        return value

        # Some languages expose declarations through
        # a declarator.
        declarator = node.child_by_field_name(
            "declarator"
        )

        if declarator is not None:
            value = self._extract_declarator_name(
                declarator,
                text,
            )

            if value:
                return value

        # Type declarations.
        type_node = node.child_by_field_name("type")

        if type_node is not None:
            value = self._node_text(
                type_node,
                text,
            )

            if value:
                return value

        # Direct identifier fallback.
        for child in node.children:
            if child.type in {
                "identifier",
                "type_identifier",
                "field_identifier",
                "property_identifier",
                "simple_identifier",
            }:
                value = self._node_text(
                    child,
                    text,
                )

                if value:
                    return value

        return None

    def _extract_declarator_name(
        self,
        node,
        text: str,
    ) -> str | None:

        if node.type in {
            "identifier",
            "type_identifier",
            "field_identifier",
            "property_identifier",
            "simple_identifier",
        }:
            return self._node_text(
                node,
                text,
            )

        name_node = node.child_by_field_name("name")

        if name_node is not None:
            value = self._node_text(
                name_node,
                text,
            )

            if value:
                return value

        for child in node.children:
            value = self._extract_declarator_name(
                child,
                text,
            )

            if value:
                return value

        return None

    @staticmethod
    def _node_text(
        node,
        text: str,
    ) -> str:
        """
        Tree-sitter uses UTF-8 byte offsets.
        Python strings use Unicode character offsets.

        Encode the source before slicing so Unicode
        characters cannot corrupt symbol extraction.
        """

        encoded = text.encode("utf-8")

        value = encoded[
            node.start_byte:
            node.end_byte
        ].decode(
            "utf-8",
            errors="replace",
        )

        return value.strip()