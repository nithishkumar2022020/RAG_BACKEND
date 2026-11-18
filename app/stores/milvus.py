import logging

from pymilvus import MilvusClient

from app.core.config import Settings

logger = logging.getLogger(__name__)
MAX_DOCUMENT_CHUNKS = 10000


def get_all_chunks(
    client: MilvusClient,
    collection: str,
    source: str,
    owner_id: str | None = None,
) -> list[dict]:
    escaped_source = source.replace("\\", "\\\\").replace('"', '\\"')
    filters = [f'document_id == "{escaped_source}"']
    if owner_id is not None:
        escaped_owner = owner_id.replace("\\", "\\\\").replace('"', '\\"')
        filters.append(f'owner_id == "{escaped_owner}"')
    iterator = client.query_iterator(
        collection_name=collection,
        batch_size=1000,
        limit=MAX_DOCUMENT_CHUNKS,
        filter=" and ".join(filters),
        output_fields=["id", "document_id", "filename", "page", "chunk_index", "text"],
    )
    chunks = []
    try:
        while batch := iterator.next():
            chunks.extend(batch)
            if len(chunks) >= MAX_DOCUMENT_CHUNKS:
                logger.warning(
                    "Whole-document chunk limit (%s) reached for document %s",
                    MAX_DOCUMENT_CHUNKS,
                    source,
                )
                break
    finally:
        iterator.close()
    return sorted(chunks, key=lambda chunk: (chunk["page"], chunk["chunk_index"]))


class MilvusStore:
    def __init__(self, settings: Settings) -> None:
        client_options = {"uri": settings.milvus_uri}
        if settings.milvus_token:
            client_options["token"] = settings.milvus_token
        self.client = MilvusClient(**client_options)
        self.collection = settings.milvus_collection
        self.dimension = settings.embedding_dimension
        self.ready = False

    def ensure_collection(self) -> None:
        if self.ready:
            return
        if not self.client.has_collection(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                dimension=self.dimension,
                primary_field_name="id",
                id_type="string",
                max_length=64,
                vector_field_name="vector",
                metric_type="COSINE",
                enable_dynamic_field=True,
            )
        self._validate_collection_schema()
        self.client.load_collection(self.collection)
        self.ready = True

    def _validate_collection_schema(self) -> None:
        description = self.client.describe_collection(self.collection)
        field_names = {field["name"] for field in description.get("fields", [])}
        required_fields = {"id", "vector"}
        missing_fields = required_fields - field_names
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise RuntimeError(
                f"Milvus collection '{self.collection}' is incompatible; "
                f"missing required fields: {missing}. Configure a new collection "
                "and re-ingest documents."
            )

    def insert(self, chunks: list[dict], vectors: list[list[float]]) -> None:
        self.ensure_collection()
        self.client.insert(
            collection_name=self.collection,
            data=[
                {**chunk, "vector": vector} for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

    @staticmethod
    def _build_doc_filter(
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        owner_id: str | None = None,
    ) -> str | None:
        ids = []
        if document_id is not None:
            ids.append(document_id)
        if document_ids:
            ids.extend(doc_id for doc_id in document_ids if doc_id and doc_id not in ids)
        filters = []
        if ids:
            escaped_list = [doc_id.replace("\\", "\\\\").replace('"', '\\"') for doc_id in ids]
            if len(escaped_list) == 1:
                filters.append(f'document_id == "{escaped_list[0]}"')
            else:
                in_clause = ", ".join(f'"{doc_id}"' for doc_id in escaped_list)
                filters.append(f"document_id in [{in_clause}]")
        if owner_id is not None:
            escaped_owner = owner_id.replace("\\", "\\\\").replace('"', '\\"')
            filters.append(f'owner_id == "{escaped_owner}"')
        return " and ".join(filters) or None

    def search(
        self,
        vector: list[float],
        limit: int,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        owner_id: str | None = None,
    ) -> list[dict]:
        self.ensure_collection()
        search_options = {}
        doc_filter = self._build_doc_filter(
            document_id=document_id,
            document_ids=document_ids,
            owner_id=owner_id,
        )
        if doc_filter is not None:
            search_options["filter"] = doc_filter
        results = self.client.search(
            collection_name=self.collection,
            data=[vector],
            limit=limit,
            output_fields=["document_id", "filename", "page", "chunk_index", "text"],
            **search_options,
        )[0]
        return [{"id": hit["id"], **hit.get("entity", {})} for hit in results]

    def all_chunks(
        self,
        document_id: str | None = None,
        document_ids: list[str] | None = None,
        owner_id: str | None = None,
    ) -> list[dict]:
        self.ensure_collection()
        expression = 'id != ""'
        doc_filter = self._build_doc_filter(
            document_id=document_id,
            document_ids=document_ids,
            owner_id=owner_id,
        )
        if doc_filter is not None:
            expression = f"{expression} and {doc_filter}"
        iterator = self.client.query_iterator(
            collection_name=self.collection,
            batch_size=1000,
            filter=expression,
            output_fields=["id", "document_id", "filename", "page", "chunk_index", "text"],
        )
        chunks = []
        try:
            while batch := iterator.next():
                chunks.extend(batch)
        finally:
            iterator.close()
        return chunks

    def get_all_chunks(self, document_id: str, owner_id: str | None = None) -> list[dict]:
        return get_all_chunks(self.client, self.collection, document_id, owner_id=owner_id)
