from sentence_transformers import CrossEncoder

from config.settings import RERANKER_MODEL, DEVICE


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
