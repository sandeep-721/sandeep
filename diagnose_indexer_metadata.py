import os
from pathlib import Path

from retrieval.indexer import Indexer
from retrieval.vector_store import VectorStore


PROJECT_ROOT = Path(
    os.environ["UNIVERSAL_RAG_PROJECT_ROOT"]
)

TARGET_FILE = (
    PROJECT_ROOT
    / "Assets"
    / "Character_VP"
    / "Scripts"
    / "Facial_Automation_Scripts"
    / "CharacterFacialProfile.cs"
)


def main():
    print("=" * 80)
    print("INDEXER STRUCTURAL METADATA TEST")
    print("=" * 80)

    indexer = Indexer(
        project_root=PROJECT_ROOT
    )

    try:
        print()
        print(f"Target: {TARGET_FILE}")
        print()

        result = indexer.index_file(
            TARGET_FILE
        )

        print()
        print(f"index_file() returned: {result}")

        print()
        print("-" * 80)
        print("QDRANT PAYLOAD VERIFICATION")
        print("-" * 80)

        store = VectorStore()

        points = store.client.scroll(
            collection_name=store.collection_name,
            scroll_filter=store.build_filter(
                source=str(TARGET_FILE.resolve())
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )[0]

        print()
        print(f"Chunks in Qdrant: {len(points)}")
        print()

        for point in points:
            payload = point.payload or {}

            print(
                f"[chunk {payload.get('chunk_index')}] "
                f"class={payload.get('class_name')} "
                f"member={payload.get('member_name')} "
                f"kind={payload.get('member_kind')} "
                f"type={payload.get('chunk_type')}"
            )

            embedding_text = payload.get(
                "embedding_text",
                ""
            )

            print(
                f"embedding_text preview: "
                f"{embedding_text[:200].replace(chr(10), ' | ')}"
            )

            print()

        print("-" * 80)
        print("TEST COMPLETE")
        print("-" * 80)

    finally:
        indexer.close()
        VectorStore.shutdown()


if __name__ == "__main__":
    main()
