from retrieval.hybrid_search import HybridSearch


class FakeStore:
    def get_adjacent_chunks(
        self,
        source,
        file_hash,
        chunk_index,
        window,
    ):
        return [
            {
                "id": "prev",
                "chunk_index": chunk_index - 1,
                "payload": {
                    "text": "previous chunk",
                },
            },
            {
                "id": "next",
                "chunk_index": chunk_index + 1,
                "payload": {
                    "text": "next chunk",
                },
            },
        ]


def test_context_expansion_preserves_anchor_and_adds_neighbors():
    hybrid = object.__new__(HybridSearch)

    class Dense:
        store = FakeStore()

    hybrid.dense = Dense()

    results = [
        {
            "id": "anchor",
            "payload": {
                "source": "example.py",
                "file_hash": "abc",
                "chunk_index": 4,
                "text": "anchor chunk",
            },
        }
    ]

    expanded = hybrid._expand_context(
        results,
        window=1,
    )

    assert expanded[0]["context_window"] == 1
    assert [
        chunk["chunk_index"]
        for chunk in expanded[0]["context_chunks"]
    ] == [3, 4, 5]
    assert [
        chunk["role"]
        for chunk in expanded[0]["context_chunks"]
    ] == ["before", "anchor", "after"]
    assert expanded[0]["expanded_context"] == (
        "previous chunk\n\nanchor chunk\n\nnext chunk"
    )


def test_context_expansion_falls_back_to_anchor():
    hybrid = object.__new__(HybridSearch)

    class Dense:
        store = FakeStore()

    hybrid.dense = Dense()

    results = [
        {
            "id": "anchor",
            "payload": {
                "text": "anchor only",
            },
        }
    ]

    expanded = hybrid._expand_context(
        results,
        window=1,
    )

    assert expanded[0]["context_window"] == 0
    assert expanded[0]["expanded_context"] == "anchor only"
