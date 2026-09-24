from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, DEVICE


# Changes whenever the embedding model or query/document
# embedding protocol changes.
EMBEDDING_VERSION = "qwen3-embedding-0.6b-query-document-v1"


class Embedder:
    _model = None

    def __init__(self):
        if Embedder._model is None:
            Embedder._model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=DEVICE,
            )

        self.model = Embedder._model

    def encode_query(
        self,
        queries: list[str],
    ) -> list[list[float]]:
        if not queries:
            return []

        embeddings = self.model.encode_query(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def encode_documents(
        self,
        documents: list[str],
    ) -> list[list[float]]:
        if not documents:
            return []

        embeddings = self.model.encode_document(
            documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    @classmethod
    def shutdown(cls):
        cls._model = None