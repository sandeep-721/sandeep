from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from retrieval.vector_store import VectorStore
from retrieval.lexical_index import LexicalIndex


def main():
    store = VectorStore()
    lexical = LexicalIndex()

    print("=== REBUILDING LEXICAL INDEX ===")

    lexical.clear()

    offset = None
    total = 0
    skipped = 0

    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            scroll_filter=None,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            payload = point.payload or {}

            if not payload.get("text"):
                skipped += 1
                continue

            lexical.upsert(
                point_id=point.id,
                payload=payload,
            )

            total += 1

        if next_offset is None:
            break

        offset = next_offset

    lexical.save()

    print()
    print("=== REBUILD COMPLETE ===")
    print(f"Qdrant points processed: {total}")
    print(f"Points skipped: {skipped}")
    print(f"Lexical documents: {lexical.count()}")

    VectorStore.shutdown()


if __name__ == "__main__":
    main()