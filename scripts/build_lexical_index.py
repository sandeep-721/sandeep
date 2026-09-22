from retrieval.lexical_index import LexicalIndex
from retrieval.vector_store import VectorStore


vector_store = VectorStore()
lexical_index = LexicalIndex()

lexical_index.clear()

offset = None
total = 0
pages = 0

while True:
    result = vector_store.client.scroll(
        collection_name=vector_store.collection_name,
        limit=1000,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )

    points, next_offset = result

    for point in points:
        lexical_index.upsert(
            point.id,
            point.payload,
        )

    total += len(points)
    pages += 1

    print(
        f"PAGE {pages} | "
        f"POINTS {len(points)} | "
        f"TOTAL {total}"
    )

    if next_offset is None:
        break

    offset = next_offset

lexical_index.save()

print()
print("LEXICAL DOCUMENTS:", lexical_index.count())
print("QDRANT POINTS:", total)

vector_store.close()
