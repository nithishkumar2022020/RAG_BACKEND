from unittest.mock import patch

from app.core.config import Settings
from app.stores.milvus import MilvusStore


def test_milvus_store_passes_zilliz_endpoint_and_token():
    settings = Settings(
        milvus_uri="https://cluster.example.zillizcloud.com",
        milvus_token="zilliz-user:secret",
        _env_file=None,
    )

    with patch("app.stores.milvus.MilvusClient") as client:
        MilvusStore(settings)

    client.assert_called_once_with(
        uri="https://cluster.example.zillizcloud.com",
        token="zilliz-user:secret",
    )


def test_milvus_store_omits_empty_token_for_local_server():
    settings = Settings(milvus_uri="http://localhost:19530", milvus_token="", _env_file=None)

    with patch("app.stores.milvus.MilvusClient") as client:
        MilvusStore(settings)

    client.assert_called_once_with(uri="http://localhost:19530")


def test_milvus_store_creates_and_loads_compatible_collection():
    settings = Settings(
        milvus_uri="https://cluster.example.zillizcloud.com",
        milvus_token="zilliz-user:secret",
        milvus_collection="rag_chunks_v2",
        _env_file=None,
    )
    with patch("app.stores.milvus.MilvusClient") as client_class:
        client = client_class.return_value
        client.has_collection.return_value = False
        client.describe_collection.return_value = {
            "fields": [{"name": "id"}, {"name": "vector"}]
        }

        store = MilvusStore(settings)
        store.ensure_collection()

    client.create_collection.assert_called_once_with(
        collection_name="rag_chunks_v2",
        dimension=768,
        primary_field_name="id",
        id_type="string",
        max_length=64,
        vector_field_name="vector",
        metric_type="COSINE",
        enable_dynamic_field=True,
    )
    client.load_collection.assert_called_once_with("rag_chunks_v2")
    assert store.ready


def test_milvus_store_rejects_legacy_collection_schema():
    settings = Settings(
        milvus_uri="https://cluster.example.zillizcloud.com",
        milvus_token="zilliz-user:secret",
        milvus_collection="rag_db_copy",
        _env_file=None,
    )
    with patch("app.stores.milvus.MilvusClient") as client_class:
        client = client_class.return_value
        client.has_collection.return_value = True
        client.describe_collection.return_value = {
            "fields": [
                {"name": "primary_key"},
                {"name": "vector"},
                {"name": "text"},
                {"name": "page"},
            ]
        }

        store = MilvusStore(settings)
        try:
            store.ensure_collection()
        except RuntimeError as error:
            assert "missing required fields: id" in str(error)
            assert "re-ingest documents" in str(error)
        else:
            raise AssertionError("Expected incompatible collection schema to fail")

        client.load_collection.assert_not_called()
        assert not store.ready
