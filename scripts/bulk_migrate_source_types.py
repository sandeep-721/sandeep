import sys
from pathlib import Path
from collections import Counter

from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


from metadata.detector import detect_source_type
from retrieval.vector_store import VectorStore


SOURCE_COLLECTION = "universal_knowledge"
TEMP_COLLECTION = "universal_knowledge_migrated"


def get_project_root(source: str) -> Path | None:
    if source.startswith(
        r"D:\AI\Universal-RAG"
    ):
        return Path(
            r"D:\AI\Universal-RAG"
        )

    if source.startswith(
        r"D:\Unity 6\Facial_Test"
    ):
        return Path(
            r"D:\Unity 6\Facial_Test"
        )

    return None


def main():
    store = VectorStore()

    client = store.client

    print("Reading collection configuration...")

    collection_info = client.get_collection(
        SOURCE_COLLECTION
    )

    vector_config = (
        collection_info.config.params.vectors
    )

    vector_size = vector_config.size
    distance = vector_config.distance

    print(
        f"Vector size: {vector_size}"
    )
    print(
        f"Distance: {distance}"
    )

    if client.collection_exists(
        TEMP_COLLECTION
    ):
        print(
            f"Removing old temporary collection: "
            f"{TEMP_COLLECTION}"
        )

        client.delete_collection(
            TEMP_COLLECTION
        )

    print(
        f"\nCreating temporary collection: "
        f"{TEMP_COLLECTION}"
    )

    client.create_collection(
        collection_name=TEMP_COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )

    offset = None
    batch = []
    total = 0
    changed = 0
    unchanged = 0

    counts = Counter()

    print("\nMigrating points...\n")

    while True:
        results = client.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        points, next_offset = results

        for point in points:
            payload = dict(
                point.payload or {}
            )

            source = payload.get(
                "source"
            )

            if source:
                project_root = (
                    get_project_root(source)
                )

                new_type = detect_source_type(
                    Path(source),
                    project_root,
                )

                old_type = payload.get(
                    "source_type"
                )

                if old_type != new_type:
                    changed += 1
                else:
                    unchanged += 1

                payload["source_type"] = (
                    new_type
                )

                counts[new_type] += 1

            vector = point.vector

            batch.append(
                PointStruct(
                    id=point.id,
                    vector=vector,
                    payload=payload,
                )
            )

            total += 1

            if len(batch) >= 500:
                client.upsert(
                    collection_name=TEMP_COLLECTION,
                    points=batch,
                )

                print(
                    f"Copied: {total}"
                )

                batch.clear()

        if next_offset is None:
            break

        offset = next_offset

    if batch:
        client.upsert(
            collection_name=TEMP_COLLECTION,
            points=batch,
        )

    print(
        "\n=== MIGRATION SUMMARY ==="
    )

    print(
        f"Total points: {total}"
    )

    print(
        f"Metadata changed: {changed}"
    )

    print(
        f"Already correct: {unchanged}"
    )

    print("\nNew source types:")

    for source_type, count in sorted(
        counts.items()
    ):
        print(
            f"{source_type}: {count}"
        )

    temp_count = client.count(
        collection_name=TEMP_COLLECTION
    ).count

    print(
        f"\nTemporary collection points: "
        f"{temp_count}"
    )

    if temp_count != total:
        raise RuntimeError(
            "Verification failed: temporary "
            "collection point count does not "
            "match source collection."
        )

    print(
        "\nMIGRATION COPY VERIFIED."
    )

    print(
        "Original collection has NOT been "
        "deleted or modified."
    )

    VectorStore.shutdown()


if __name__ == "__main__":
    main()