from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_MODEL, DEVICE


class Embedder:
    _model = None

    def __init__(self):
        if Embedder._model is None:
            Embedder._model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=DEVICE,
            )

        self.model = Embedder._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    @classmethod
    def shutdown(cls):
        cls._model = None