from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, DEVICE


def test_embedding_model():
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=DEVICE,
    )

    assert model.get_embedding_dimension() > 0

    embedding = model.encode(
        "Universal RAG test",
        convert_to_tensor=True,
    )

    assert embedding.shape[-1] == model.get_embedding_dimension()
    assert embedding.numel() > 0
