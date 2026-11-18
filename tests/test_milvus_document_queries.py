from unittest.mock import patch

from app.core.config import Settings
from app.stores.milvus import MilvusStore, get_all_chunks


class FakeIterator:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.closed = False

    def next(self):
        return next(self.batches, [])

    def close(self):
        self.closed = True


def test_get_all_chunks_filters_and_sorts_by_page_and_chunk():
    rows = [
        {"id": "b", "document_id": "doc-1", "page": 2, "chunk_index": 0},
        {"id": "a", "document_id": "doc-1", "page": 1, "chunk_index": 1},
        {"id": "z", "document_id": "doc-1", "page": 1, "chunk_index": 0},
    ]
    iterator = FakeIterator([rows])
    with patch("app.stores.milvus.MilvusClient") as client_class:
        client = client_class.return_value
        client.query_iterator.return_value = iterator
        chunks = get_all_chunks(client, "rag_chunks_v2", "doc-1")

    assert [chunk["id"] for chunk in chunks] == ["z", "a", "b"]
    client.query_iterator.assert_called_once_with(
        collection_name="rag_chunks_v2",
        batch_size=1000,
        limit=10000,
        filter='document_id == "doc-1"',
        output_fields=["id", "document_id", "filename", "page", "chunk_index", "text"],
    )
    assert iterator.closed


def test_vector_search_filters_by_selected_document():
    settings = Settings(
        milvus_uri="http://localhost:19530",
        milvus_token="",
        _env_file=None,
    )
    with patch("app.stores.milvus.MilvusClient") as client_class:
        client = client_class.return_value
        client.has_collection.return_value = True
        client.describe_collection.return_value = {
            "fields": [{"name": "id"}, {"name": "vector"}]
        }
        client.search.return_value = [[{"id": "chunk-1", "entity": {"text": "excerpt"}}]]
        store = MilvusStore(settings)
        store.search([0.1, 0.2], 5, document_id='doc"1')

    client.search.assert_called_once()
    kwargs = client.search.call_args.kwargs
    assert kwargs["filter"] == 'document_id == "doc\\"1"'
    assert kwargs["output_fields"] == ["document_id", "filename", "page", "chunk_index", "text"]


def test_vector_search_scopes_results_to_authenticated_owner():
    settings = Settings(_env_file=None)
    with patch("app.stores.milvus.MilvusClient") as client_class:
        client = client_class.return_value
        client.has_collection.return_value = True
        client.describe_collection.return_value = {
            "fields": [{"name": "id"}, {"name": "vector"}]
        }
        client.search.return_value = [[{"id": "chunk-1", "entity": {"text": "excerpt"}}]]
        store = MilvusStore(settings)
        store.search([0.1, 0.2], 5, document_id="doc-1", owner_id='user"123')

    assert client.search.call_args.kwargs["filter"] == (
        'document_id == "doc-1" and owner_id == "user\\"123"'
    )


def test_all_chunks_query_scopes_results_to_authenticated_owner():
    iterator = FakeIterator([[]])
    with patch("app.stores.milvus.MilvusClient") as client_class:
        client = client_class.return_value
        client.query_iterator.return_value = iterator
        get_all_chunks(client, "rag_chunks_v2", "doc-1", owner_id="user-1")

    assert client.query_iterator.call_args.kwargs["filter"] == (
        'document_id == "doc-1" and owner_id == "user-1"'
    )
