from generation.claim_verifier import ClaimVerifier
from generation.evidence_packet import (
    EvidenceItem,
    EvidencePacket,
)


def _packet():
    return EvidencePacket(
        query="What does Example do?",
        evidence=(
            EvidenceItem(
                evidence_id="E1",
                rank=1,
                source="example.py",
                file_hash="abc",
                chunk_index=4,
                chunk_start=100,
                chunk_end=150,
                text=(
                    "The Example class loads the project "
                    "configuration and validates the input."
                ),
                expanded_context=(
                    "The Example class loads the project "
                    "configuration and validates the input."
                ),
                context_window=0,
            ),
        ),
        sources=(),
        max_chars=4000,
    )


def test_valid_citation_is_verified():
    result = ClaimVerifier().verify(
        "The Example class loads the project configuration. [E1]",
        _packet(),
    )

    assert result.verified
    assert result.citation_coverage == 1.0
    assert result.invalid_citations == ()
    assert result.uncited_claims == ()
    assert result.weak_support == ()
    assert result.supports[0].evidence_id == "E1"
    assert result.supports[0].score > 0


def test_uncited_claim_is_detected():
    result = ClaimVerifier().verify(
        "The Example class loads the project configuration.",
        _packet(),
    )

    assert not result.verified
    assert result.citation_coverage == 0.0
    assert result.uncited_claims == ("C1",)


def test_invalid_citation_is_detected():
    result = ClaimVerifier().verify(
        "The Example class loads the project configuration. [E99]",
        _packet(),
    )

    assert not result.verified
    assert result.invalid_citations == ("E99",)
    assert result.uncited_claims == ("C1",)

def _rag_with_fake_components(answer):
    from rag import RAG
    from generation.claim_verifier import ClaimVerifier
    from generation.context_builder import ContextBuilder

    class FakeSearch:
        def search(self, **kwargs):
            return [
                {
                    "id": "example",
                    "score": 0.9,
                    "final_rerank_score": 1.2,
                    "payload": {
                        "source": "example.py",
                        "file_hash": "abc",
                        "chunk_index": 4,
                        "chunk_start": 100,
                        "chunk_end": 150,
                        "text": (
                            "The Example class loads the project "
                            "configuration."
                        ),
                        "expanded_context": (
                            "The Example class loads the project "
                            "configuration."
                        ),
                        "context_window": 0,
                        "project": "Demo",
                        "language": "Python",
                        "symbol": "Example",
                        "symbol_kind": "class",
                    },
                }
            ]

    class FakeLLM:
        def generate(self, **kwargs):
            return answer

    rag = object.__new__(RAG)
    rag.search = FakeSearch()
    rag.context_builder = ContextBuilder()
    rag.claim_verifier = ClaimVerifier()
    rag.llm = FakeLLM()
    rag.candidate_limit = 30
    rag.result_limit = 8
    rag.project_root = None

    return rag


def test_rag_ask_returns_verified_grounding_result():
    rag = _rag_with_fake_components(
        "The Example class loads the project configuration. [E1]"
    )

    result = rag.ask(
        "What does Example do?"
    )

    assert result["verification"]["citation_coverage"] == 1.0
    assert result["verification"]["invalid_citations"] == []
    assert result["verification"]["uncited_claims"] == []
    assert result["evidence_packet"]["query"] == (
        "What does Example do?"
    )


def test_rag_ask_reports_uncited_claim():
    rag = _rag_with_fake_components(
        "The Example class loads the project configuration."
    )

    result = rag.ask(
        "What does Example do?"
    )

    assert result["verification"]["verified"] is False
    assert result["verification"]["citation_coverage"] == 0.0
    assert result["verification"]["uncited_claims"] == ["C1"]

def test_rag_ask_uses_grounding_repair_when_new_evidence_improves_result():
    from rag import RAG
    from generation.claim_verifier import ClaimVerifier
    from generation.context_builder import ContextBuilder

    class FakeSearch:
        def __init__(self):
            self.calls = 0

        def search(self, **kwargs):
            self.calls += 1

            if self.calls == 1:
                text = (
                    "The Example class loads the configuration "
                    "from project settings."
                )
                chunk_index = 1
            else:
                text = (
                    "The Example class validates the project "
                    "configuration before loading it."
                )
                chunk_index = 2

            return [
                {
                    "id": f"example-{self.calls}",
                    "score": 0.9,
                    "final_rerank_score": 1.2,
                    "payload": {
                        "source": f"example_{self.calls}.py",
                        "file_hash": f"hash-{self.calls}",
                        "chunk_index": chunk_index,
                        "chunk_start": 1,
                        "chunk_end": 20,
                        "text": text,
                        "expanded_context": text,
                        "context_window": 0,
                        "project": "Demo",
                        "language": "Python",
                        "symbol": "Example",
                        "symbol_kind": "class",
                    },
                }
            ]

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1

            if self.calls == 1:
                return (
                    "The Example class validates the project "
                    "configuration."
                )

            return (
                "The Example class validates the project "
                "configuration before loading it. [E2]"
            )

    rag = object.__new__(RAG)
    rag.search = FakeSearch()
    rag.context_builder = ContextBuilder()
    rag.claim_verifier = ClaimVerifier()
    rag.llm = FakeLLM()
    rag.candidate_limit = 30
    rag.result_limit = 8
    rag.project_root = None

    result = rag.ask(
        "What does Example do?"
    )

    assert result["repair"]["attempts"] == 1
    assert result["repair"]["performed"] is True
    assert result["repair"]["used"] is True
    assert result["repair"]["added_results"] == 1
    assert "[E2]" in result["answer"]


def test_rag_ask_does_not_use_worse_repair():
    from rag import RAG
    from generation.claim_verifier import ClaimVerifier
    from generation.context_builder import ContextBuilder

    class FakeSearch:
        def __init__(self):
            self.calls = 0

        def search(self, **kwargs):
            self.calls += 1

            suffix = "initial" if self.calls == 1 else "repair"

            return [
                {
                    "id": f"example-{suffix}",
                    "score": 0.9,
                    "final_rerank_score": 1.2,
                    "payload": {
                        "source": f"{suffix}.py",
                        "file_hash": suffix,
                        "chunk_index": self.calls,
                        "chunk_start": 1,
                        "chunk_end": 20,
                        "text": (
                            "The Example class loads the "
                            "project configuration."
                        ),
                        "expanded_context": (
                            "The Example class loads the "
                            "project configuration."
                        ),
                        "context_window": 0,
                        "project": "Demo",
                        "language": "Python",
                        "symbol": "Example",
                        "symbol_kind": "class",
                    },
                }
            ]

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1

            if self.calls == 1:
                return (
                    "The Example class loads the project "
                    "configuration."
                )

            return (
                "The Example class does something unrelated. [E99]"
            )

    rag = object.__new__(RAG)
    rag.search = FakeSearch()
    rag.context_builder = ContextBuilder()
    rag.claim_verifier = ClaimVerifier()
    rag.llm = FakeLLM()
    rag.candidate_limit = 30
    rag.result_limit = 8
    rag.project_root = None

    result = rag.ask(
        "What does Example do?"
    )

    assert result["repair"]["performed"] is True
    assert result["repair"]["used"] is False
    assert result["answer"] == (
        "The Example class loads the project configuration."
    )
    assert result["verification"]["verified"] is False
    assert result["verification"]["uncited_claims"] == ["C1"]
