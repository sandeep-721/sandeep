import os
from pathlib import Path

from ingestion.reader import read_text_file
from ingestion.chunker import (
    chunk_text,
)
from retrieval.indexer import build_symbol_metadata
from metadata.symbols import UniversalSymbolAnalyzer


PROJECT_ROOT = Path(
    os.environ["UNIVERSAL_RAG_PROJECT_ROOT"]
)

PROJECT_FILE = (
    PROJECT_ROOT
    / "Assets"
    / "Character_VP"
    / "Scripts"
    / "Facial_Automation_Scripts"
    / "CharacterFacialProfile.cs"
)


def main():
    text = read_text_file(PROJECT_FILE)

    chunks = chunk_text(text, extension=".cs")

    structures = UniversalSymbolAnalyzer().analyze_file(PROJECT_FILE)

    print()
    print("=" * 80)
    print("STRUCTURAL METADATA TEST")
    print("=" * 80)

    print(
        f"Structures found: {len(structures)}"
    )

    print(
        f"Chunks found: {len(chunks)}"
    )

    print()
    print("-" * 80)

    for index, chunk in enumerate(
        chunks
    ):
        metadata = build_symbol_metadata(
            symbols=structures,
            chunk_start=chunk.start,
            chunk_end=chunk.end,
        )

        class_name = metadata["class_name"]
        member_name = metadata["member_name"]
        member_kind = metadata["member_kind"]

        preview = (
            chunk.text[:100]
            .replace("\n", " ")
            .replace("\r", " ")
        )

        print(
            f"[{index}] "
            f"class={class_name} "
            f"member={member_name} "
            f"kind={member_kind}"
        )

        print(
            f"     {preview}"
        )

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()







