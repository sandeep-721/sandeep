from sentence_transformers import CrossEncoder

from config.settings import RERANKER_MODEL, DEVICE
from reranking.reranker import Reranker


def test_reranker_ranks_relevant_document_higher():
    model = CrossEncoder(
        RERANKER_MODEL,
        device=DEVICE,
    )

    query = "What device does the Universal RAG use?"

    positive_document = (
        "The Universal RAG uses Qdrant as its vector database. "
        "Qdrant provides local vector storage and similarity search."
    )

    negative_document = (
        "The facial automation project uses Unity 6 with URP "
        "for rendering a character and controlling facial animation."
    )

    scores = model.predict(
        [
            (query, positive_document),
            (query, negative_document),
        ],
        show_progress_bar=False,
    )

    assert float(scores[0]) > float(scores[1])


def test_source_authority_is_history_aware():
    assert Reranker._source_authority(
        {"source_type": "project_documentation"}
    ) > Reranker._source_authority(
        {"source_type": "project_history"}
    )

    assert Reranker._source_authority(
        {"source_type": "project_history"}
    ) > Reranker._source_authority(
        {"source_type": "project_test"}
    )

    assert Reranker._source_authority(
        {"source_type": "project_configuration"}
    ) > Reranker._source_authority(
        {"source_type": "project_code"}
    )

    assert Reranker._source_authority(
        {"source_type": "rag_documentation"}
    ) > Reranker._source_authority(
        {"source_type": "rag_history"}
    )

    assert Reranker._source_authority(
        {"source_type": "third_party_code"}
    ) < Reranker._source_authority(
        {"source_type": "project_code"}
    )
