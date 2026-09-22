import sys
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metadata.detector import detect_source_type
from retrieval.vector_store import VectorStore


BATCH_SIZE = 250


def get_project_root(source: str) -> Path | None:
    if source.startswith(r"D:\AI\Universal-RAG"):
        return Path(r"D:\AI\Universal-RAG")

    if source.startswith(r"D:\Unity 6\Facial_Test"):
        return Path(r"D:\Unity 6\Facial_Test")

    return None


def flush_updates(store: VectorStore, pending_updates: dict[str, list]):
    """
    Apply pending source_type updates in batches.

    pending_updates:
        {
            "rag_code": [point_id, point_id, ...],
            "project_code": [point_id, point_id, ...],
            ...
        }
    """

    updated = 0

    for source_type, point_ids in pending_updates.items():
        if not point_ids:
            continue

        for start in range(0, len(point_ids), BATCH_SIZE):
            batch = point_ids[start:start + BATCH_SIZE]

            store.client.set_payload(
                collection_name=store.collection_name,
                payload={
                    "source_type": source_type,
                },
                points=batch,
            )

            updated += len(batch)

            print(
                f"  Updated {len(batch)} chunks -> "
                f"{source_type} "
                f"(batch {start // BATCH_SIZE + 1})",
                flush=True,
            )

    pending_updates.clear()

    return updated


def migrate_source_types(store: VectorStore):
    counts = Counter()

    pending_updates = defaultdict(list)

    total = 0
    changed = 0
    skipped = 0
    missing_source = 0

    offset = None

    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            break

        print(
            f"\nScanning batch: {total + 1}"
            f"-{total + len(points)}",
            flush=True,
        )

        for point in points:
            total += 1

            payload = point.payload or {}

            source = payload.get("source")

            if not source:
                missing_source += 1
                continue

            project_root = get_project_root(source)

            new_type = detect_source_type(
                Path(source),
                project_root,
            )

            old_type = payload.get("source_type")

            counts[
                f"{old_type or 'MISSING'} -> {new_type}"
            ] += 1

            if old_type == new_type:
                skipped += 1
                continue

            pending_updates[new_type].append(point.id)
            changed += 1

            pending_count = sum(
                len(ids)
                for ids in pending_updates.values()
            )

            if pending_count >= BATCH_SIZE:
                updated_now = flush_updates(
                    store,
                    pending_updates,
                )

                print(
                    f"  Flushed {updated_now} pending updates.",
                    flush=True,
                )

        if next_offset is None:
            break

        offset = next_offset

    if pending_updates:
        updated_now = flush_updates(
            store,
            pending_updates,
        )

        print(
            f"\n  Flushed final {updated_now} updates.",
            flush=True,
        )

    return (
        total,
        counts,
        changed,
        skipped,
        missing_source,
    )


def main():
    store = VectorStore()

    try:
        (
            total,
            counts,
            changed,
            skipped,
            missing_source,
        ) = migrate_source_types(store)

        print("\n=== SOURCE TYPE MIGRATION ===\n")

        for key, count in sorted(counts.items()):
            print(f"{key}: {count}")

        print(f"\nTotal scanned: {total}")
        print(f"Updated: {changed}")
        print(f"Already correct: {skipped}")
        print(f"Missing source: {missing_source}")

        print("\nSOURCE TYPE MIGRATION COMPLETE.")

    finally:
        VectorStore.shutdown()


if __name__ == "__main__":
    main()