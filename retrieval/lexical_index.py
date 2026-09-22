import json
import re

from config.settings import DATA_DIR


LEXICAL_DIR = DATA_DIR / "lexical"
LEXICAL_FILE = LEXICAL_DIR / "documents.json"


class LexicalIndex:
    """
    Persistent lexical index used by the sparse/BM25 retriever.

    The tokenizer is code-aware and understands:
        - normal words
        - PascalCase
        - camelCase
        - ALL_CAPS
        - snake_case
        - kebab-case
        - dotted identifiers
        - filenames
        - symbols
        - numbers
    """

    VERSION = 3

    def __init__(self):
        LEXICAL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.documents = {}
        self.load()

    @staticmethod
    def _split_identifier(value: str) -> list[str]:
        if not value:
            return []

        value = str(value)

        parts = re.split(
            r"[^A-Za-z0-9]+",
            value,
        )

        tokens = []

        for part in parts:
            if not part:
                continue

            # Preserve the complete identifier.
            tokens.append(part.lower())

            # Split PascalCase / camelCase / acronyms / digits.
            camel_parts = re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|"
                r"[A-Z]?[a-z]+|"
                r"[0-9]+",
                part,
            )

            for component in camel_parts:
                if component:
                    component = component.lower()

                    # Avoid duplicating the complete token.
                    if component != part.lower():
                        tokens.append(component)

        return tokens

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """
        Tokenize arbitrary text while preserving useful programming
        identifiers.

        Example:

            LateUpdateBoneDriver

        becomes:

            lateupdatebonedriver
            late
            update
            bone
            driver
        """

        if not text:
            return []

        tokens = []

        raw_tokens = re.findall(
            r"[A-Za-z_][A-Za-z0-9_.-]*|[0-9]+",
            text,
        )

        for raw_token in raw_tokens:
            tokens.extend(
                cls._split_identifier(raw_token)
            )

        return tokens

    @classmethod
    def _build_lexical_text(
        cls,
        payload: dict,
    ) -> str:
        """
        Build the lexical representation used by BM25.

        Structural metadata is deliberately included so code search
        can match filenames, symbols, classes and methods even when
        those identifiers are not prominent in the chunk text.
        """

        fields = [
            payload.get("source"),
            payload.get("path"),
            payload.get("symbol"),
            payload.get("symbol_kind"),
            payload.get("class_name"),
            payload.get("member_name"),
            payload.get("member_kind"),
            payload.get("chunk_type"),
            payload.get("language"),
            payload.get("software"),
            payload.get("software_version"),
            payload.get("project"),
            payload.get("text"),
        ]

        return " ".join(
            str(value)
            for value in fields
            if value
        )

    def load(self):
        if not LEXICAL_FILE.exists():
            self.documents = {}
            return

        with LEXICAL_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        version = data.get("version")

        if version == self.VERSION:
            self.documents = data.get(
                "documents",
                {},
            )
            return

        if version in {1, 2}:
            old_documents = data.get(
                "documents",
                {},
            )

            self.documents = {}

            for point_id, document in old_documents.items():
                payload = document.get(
                    "payload",
                    {},
                )

                text = payload.get(
                    "text",
                    document.get(
                        "text",
                        "",
                    ),
                )

                if not text:
                    continue

                lexical_text = self._build_lexical_text(
                    payload
                )

                self.documents[str(point_id)] = {
                    "id": document.get(
                        "id",
                        point_id,
                    ),
                    "text": lexical_text,
                    "payload": payload,
                    "tokens": self.tokenize(
                        lexical_text
                    ),
                }

            self.save()
            return

        raise ValueError(
            f"Unsupported lexical index version: {version}"
        )

    def save(self):
        temporary_file = LEXICAL_FILE.with_suffix(
            ".tmp"
        )

        data = {
            "version": self.VERSION,
            "documents": self.documents,
        }

        with temporary_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
            )

        temporary_file.replace(
            LEXICAL_FILE
        )

    def upsert(
        self,
        point_id,
        payload: dict,
    ):
        text = payload.get(
            "text",
            "",
        )

        if not text:
            return

        lexical_text = self._build_lexical_text(
            payload
        )

        self.documents[str(point_id)] = {
            "id": point_id,
            "text": lexical_text,
            "payload": payload,
            "tokens": self.tokenize(
                lexical_text
            ),
        }

    def delete(
        self,
        point_id,
    ):
        self.documents.pop(
            str(point_id),
            None,
        )

    def delete_file(
        self,
        source: str,
    ):
        self.documents = {
            key: document
            for key, document in self.documents.items()
            if document.get(
                "payload",
                {},
            ).get("source") != source
        }

    def clear(self):
        self.documents = {}

    def count(self) -> int:
        return len(
            self.documents
        )