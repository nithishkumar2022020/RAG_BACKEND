# PDF RAG API

FastAPI service for PDF ingestion and cited question answering. It uses PyMuPDF, Gemini embeddings and generation, Milvus, BM25, and reciprocal-rank fusion.

## Run with Docker

```sh
cp .env.example .env
# Set GEMINI_API_KEY in .env
docker compose up --build -d
```

This starts Nginx on `http://localhost`, which proxies requests to the private FastAPI container. Open `http://localhost/docs` to explore the API, upload PDFs with `POST /documents`, and ask questions with `POST /questions`. Uploads are limited to 20 MiB by default. Locally stored originals use the persistent `app_data` Docker volume; chunks and vectors are stored in Milvus or Zilliz Cloud. FastAPI's port is not published directly to the host.

The Nginx proxy supports streaming responses from `POST /questions/stream` and allows uploads up to 21 MiB to accommodate the API's 20 MiB limit. This Compose configuration provides HTTP only; for public EC2 deployments, terminate HTTPS at an HTTPS-enabled load balancer or configure TLS for Nginx, and restrict inbound traffic with security groups.

For incremental answers, send the same JSON body to `POST /questions/stream`:
`{"question":"...","document_id":"optional-selected-document-id"}`. The response is
`text/event-stream` with `answer_chunk` events (`{"text":"..."}`), followed by a `citations`
event (`{"citations":[...]}`). Stream-time failures are sent as an `error` event.

For hosted Zilliz Cloud, set `MILVUS_URI` to the cluster's **public endpoint** (including `https://`), `MILVUS_TOKEN` to its cluster token, and `MILVUS_COLLECTION=rag_chunks_v2` in `.env`. The token should be kept secret. The Compose default uses the configured endpoint and does not start local Milvus. The application creates this collection with its required `id` and `vector` fields and stores chunk metadata as dynamic fields. Existing collections with a different schema are not modified; re-upload PDFs after switching to the new collection.

To run the local Milvus stack instead, set `MILVUS_URI=http://milvus:19530` in `.env` and start the profile:

```sh
docker compose --profile local-milvus up --build -d
```

For running the API directly on the host instead of in Docker, use `MILVUS_URI=http://localhost:19530` for local Milvus and follow the Python setup steps:

```sh
.venv/bin/pip install -e '.[dev]'
.venv/bin/uvicorn app.main:app --reload
```

Use `docker compose logs -f api nginx` to view API and proxy logs and `docker compose down` to stop the containers. The named `app_data` volume is retained when containers stop.

If `POST /questions` returns 503, inspect the API container logs for the underlying traceback. The response intentionally remains generic and does not expose provider errors. A Zilliz collection schema mismatch is one possible cause; use a fresh collection such as `rag_chunks_v2`, restart/rebuild the API so it reads the updated `.env`, and upload the PDFs again. The legacy collections are not modified or migrated.

Run tests with `.venv/bin/pytest`.

PDFs use semantic chunking by default: adjacent sentences are grouped by Gemini embedding
similarity, with `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `SEMANTIC_BREAKPOINT_PERCENTILE` controlling
boundaries. Set `CHUNKING_STRATEGY=fixed` in `.env` to use the original fixed-size chunking.
Embedding inputs are sent in batches of up to 100 and scheduled at 90 requests per minute by
default to stay below the observed Gemini free-tier request limit. Both values are configurable
with `EMBEDDING_BATCH_SIZE` (maximum 100) and `EMBEDDING_REQUESTS_PER_MINUTE` (maximum 99).
Quota exhaustion that still occurs returns HTTP 429; wait for the quota window to reset or review
Gemini rate limits and billing. Larger quotas may be available depending on the Gemini project and
billing plan.

To store original PDFs in Cloudflare R2, set `DOCUMENT_STORAGE_BACKEND=r2` and provide `R2_ACCOUNT_ID`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY` in `.env`. R2 credentials should be scoped to the target bucket. Objects are written below `R2_PREFIX` (default `documents`) using the generated document ID as the filename. Chunk embeddings and metadata continue to go to Milvus or Zilliz Cloud. Keep `.env` out of source control; it is excluded from the Docker build context.

For document-scoped summaries and evaluations, the existing `POST /questions` request accepts an
optional `document_id` alongside the required `question`. General and judgment-style questions use
all chunks from the selected document (up to 60,000 context characters); larger documents fall
back to the top retrieved chunks. The response keeps the existing `answer` and `citations` fields.
The API currently has no authenticated user identity or per-user chunk metadata, so retrieval is
scoped by `document_id`, not `user_id`.

PyMuPDF is AGPL-3.0 or commercially licensed; review its terms before distributing a closed-source derivative.

## Documentation

A detailed project overview, architecture notes, API flow, and implementation details are available in [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).

## Continuous integration

GitHub Actions runs Ruff and the pytest suite on pull requests targeting `main` and on pushes to `main`. To require these checks before merging, enable branch protection for `main` in GitHub and require the **Python checks** status check.
