from server.response_builder import build_ask_response


class FakeConfig:
    provider = "local_transformers"
    model = "example-model"
    base_url = None


def test_build_ask_response_preserves_existing_fields():
    result = build_ask_response(
        query="What does Example do?",
        project="Demo",
        result={
            "answer": "Example loads configuration. [E1]",
            "sources": [{"source": "example.py"}],
            "results": [{"id": "1"}],
            "evidence_packet": {
                "query": "What does Example do?",
            },
            "verification": {
                "verified": True,
                "citation_coverage": 1.0,
            },
            "repair": {
                "attempts": 0,
                "performed": False,
                "used": False,
                "added_results": 0,
            },
        },
        config=FakeConfig(),
    )

    assert result["query"] == "What does Example do?"
    assert result["project"] == "Demo"
    assert result["answer"] == (
        "Example loads configuration. [E1]"
    )
    assert result["sources"] == [
        {"source": "example.py"}
    ]
    assert result["results"] == [{"id": "1"}]
    assert result["evidence_packet"]["query"] == (
        "What does Example do?"
    )
    assert result["verification"]["verified"] is True
    assert result["repair"]["attempts"] == 0
    assert result["model"]["provider"] == (
        "local_transformers"
    )


def test_build_ask_response_supplies_grounding_defaults():
    result = build_ask_response(
        query="test",
        project=None,
        result={},
        config=FakeConfig(),
    )

    assert result["evidence_packet"] is None
    assert result["verification"] is None
    assert result["repair"] == {
        "attempts": 0,
        "performed": False,
        "used": False,
        "added_results": 0,
    }
