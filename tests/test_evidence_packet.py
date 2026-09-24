from generation.context_builder import ContextBuilder


def _result(
    source="example.py",
    chunk_index=4,
    text="anchor chunk",
):
    return {
        "id": "anchor",
        "score": 0.9,
        "final_rerank_score": 1.2,
        "payload": {
            "source": source,
            "file_hash": "abc",
            "chunk_index": chunk_index,
            "chunk_start": 100,
            "chunk_end": 150,
            "text": text,
            "expanded_context": (
                "previous chunk\n\n"
                "anchor chunk\n\n"
                "next chunk"
            ),
            "context_window": 1,
            "context_chunks": [
                {
                    "chunk_index": 3,
                    "role": "before",
                    "payload": {"text": "previous chunk"},
                },
                {
                    "chunk_index": 4,
                    "role": "anchor",
                    "payload": {"text": "anchor chunk"},
                },
                {
                    "chunk_index": 5,
                    "role": "after",
                    "payload": {"text": "next chunk"},
                },
            ],
            "project": "Demo",
            "language": "Python",
            "symbol": "Example",
            "symbol_kind": "class",
        },
    }


def test_build_packet_preserves_ranked_evidence():
    packet = ContextBuilder(max_chars=4000).build_packet(
        [_result()],
        query="What does Example do?",
    )

    assert packet.query == "What does Example do?"
    assert len(packet.evidence) == 1

    item = packet.evidence[0]

    assert item.evidence_id == "E1"
    assert item.rank == 1
    assert item.source == "example.py"
    assert item.file_hash == "abc"
    assert item.chunk_index == 4
    assert item.context_window == 1
    assert item.expanded_context == (
        "previous chunk\n\n"
        "anchor chunk\n\n"
        "next chunk"
    )
    assert item.metadata["symbol"] == "Example"
    assert item.scores["final_rerank_score"] == 1.2
    assert "[E1]" in packet.text


def test_build_packet_groups_sources():
    packet = ContextBuilder(max_chars=10000).build_packet(
        [
            _result(source="a.py", chunk_index=1),
            _result(source="a.py", chunk_index=2),
            _result(source="b.py", chunk_index=3),
        ],
        query="test",
    )

    assert len(packet.sources) == 2
    assert packet.sources[0].source == "a.py"
    assert packet.sources[0].evidence_ids == ("E1", "E2")
    assert packet.sources[0].chunks == (1, 2)

    assert packet.sources[1].source == "b.py"
    assert packet.sources[1].evidence_ids == ("E3",)
    assert packet.sources[1].chunks == (3,)


def test_build_packet_empty_results():
    packet = ContextBuilder().build_packet(
        [],
        query="nothing",
    )

    assert packet.is_empty()
    assert packet.text == ""
    assert packet.sources == ()
    assert packet.query == "nothing"
