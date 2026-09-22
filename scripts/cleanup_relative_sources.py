from qdrant_client.models import PointIdsList

from retrieval.vector_store import VectorStore
from retrieval.lexical_index import LexicalIndex


STALE_SOURCES = {
    r"config\settings.py",
    r"ingestion\reader.py",
}


def main():
    store = VectorStore()
    lexical = LexicalIndex()

    print("Starting relative-source cleanup...")

    removed_qdrant = 0
    removed_lexical = 0

    # Remove stale Qdrant entries.
    for source in sorted(STALE_SOURCES):
        points, _ = store.client.scroll(
            collection_name=store.collection_name,
            scroll_filter=store.build_filter(source=source),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )

        point_ids = [point.id for point in points]

        if point_ids:
            store.client.delete(
                collection_name=store.collection_name,
                points_selector=PointIdsList(
                    points=point_ids
                ),
            )

            removed_qdrant += len(point_ids)

            for point_id in point_ids:
                print(
                    f"Removed Qdrant point: "
                    f"{point_id} -> {source}"
                )

    # Remove the same stale documents from BM25.
    stale_ids = set()

    for document_id, document in lexical.documents.items():
        payload = document.get("payload", {})
        source = payload.get("source")

        if source in STALE_SOURCES:
            stale_ids.add(document_id)

    for document_id in stale_ids:
        del lexical.documents[document_id]
        removed_lexical += 1

        print(
            f"Removed lexical document: "
            f"{document_id}"
        )

    lexical.save()

    print()
    print("CLEANUP COMPLETE")
    print("Qdrant removed:", removed_qdrant)
    print("Lexical removed:", removed_lexical)


if __name__ == "__main__":
    main()