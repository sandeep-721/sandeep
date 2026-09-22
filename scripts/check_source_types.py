import sys
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


from retrieval.vector_store import VectorStore


def main():
    store = VectorStore()

    offset = None
    counts = Counter()
    total = 0

    while True:
        results = store.client.scroll(
            collection_name=store.collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points, next_offset = results

        total += len(points)

        for point in points:
            payload = point.payload or {}

            source_type = payload.get(
                "source_type",
                "MISSING",
            )

            counts[source_type] += 1

        if next_offset is None:
            break

        offset = next_offset

    print("\n=== SOURCE TYPE STATUS ===\n")

    print(f"Total chunks: {total}")

    for source_type, count in sorted(
        counts.items()
    ):
        print(
            f"{source_type}: {count}"
        )

    VectorStore.shutdown()


if __name__ == "__main__":
    main()