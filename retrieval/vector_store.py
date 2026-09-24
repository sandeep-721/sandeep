from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from config.settings import DATA_DIR


class VectorStore:
    _client = None

    def __init__(
        self,
        collection_name="universal_knowledge",
        vector_size=1024,
    ):
        self.collection_name = collection_name

        if VectorStore._client is None:
            VectorStore._client = QdrantClient(
                path=str(DATA_DIR / "qdrant")
            )

        self.client = VectorStore._client

        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size):
        collections = (
            self.client
            .get_collections()
            .collections
        )

        exists = any(
            collection.name == self.collection_name
            for collection in collections
        )

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def build_filter(
        self,
        software=None,
        software_version=None,
        project=None,
        content_type=None,
        language=None,
        human_language=None,
        file_hash=None,
        source=None,
        embedding_version=None,
    ):
        """
        Build an optional metadata filter.

        `language` refers to the programming/document
        language, for example:
            csharp
            python
            javascript
            markdown

        `human_language` refers to the natural language
        detected in the content, for example:
            en
            te
            ta
            hi

        `embedding_version` identifies the embedding model/
        protocol used to create the indexed vector.
        """

        conditions = []

        filters = {
            "software": software,
            "software_version": software_version,
            "project": project,
            "content_type": content_type,
            "language": language,
            "human_language": human_language,
            "file_hash": file_hash,
            "source": source,
            "embedding_version": embedding_version,
        }

        for field, value in filters.items():
            if value is not None:
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(
                            value=value
                        ),
                    )
                )

        if not conditions:
            return None

        return Filter(
            must=conditions
        )

    def file_exists(
        self,
        file_hash,
        embedding_version=None,
    ):
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self.build_filter(
                file_hash=file_hash,
                embedding_version=embedding_version,
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )

        return bool(results[0])

    def get_indexed_sources(self):
        sources = set()
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                source = point.payload.get(
                    "source"
                )

                if source:
                    sources.add(source)

            if next_offset is None:
                break

            offset = next_offset

        return sources

    def upsert(
        self,
        ids,
        vectors,
        payloads,
    ):
        points = [
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            for point_id, vector, payload in zip(
                ids,
                vectors,
                payloads,
            )
        ]

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def delete_file(self, source):
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="source",
                    match=MatchValue(
                        value=source
                    ),
                )
            ]
        )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=search_filter,
        )

    def count(self):
        return self.client.count(
            collection_name=self.collection_name
        ).count

    def close(self):
        return

    @classmethod
    def shutdown(cls):
        if cls._client is not None:
            client = cls._client
            cls._client = None

            try:
                client.close()
            except Exception:
                pass