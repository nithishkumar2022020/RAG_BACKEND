# PDF RAG API Project Documentation

## 1. Overview

This project is a FastAPI-based Retrieval-Augmented Generation (RAG) service focused on PDF documents. It allows a user to upload a PDF, extract text page by page, split that text into chunks, embed those chunks with Gemini, store them in Milvus, and then answer natural-language questions grounded in the uploaded document content.

The implementation follows a practical hybrid retrieval pattern:

- Dense retrieval using vector similarity in Milvus
- Lexical retrieval using BM25 over chunk text
- Reciprocal Rank Fusion (RRF) to combine both retrieval streams
- Final answer generation with Gemini using only the retrieved text snippets as source context

This project is a working backend foundation for PDF-based Q&A and can be extended with multi-document support, persistence layers, authentication, and front-end integrations.

## 2. Current status

The project is in a working MVP state. The implemented features include:

- PDF upload validation and parsing
- Local storage of uploaded original PDFs
- Text chunking with configurable overlap
- Embedding generation through Gemini
- Milvus-backed vector indexing
- Hybrid search with BM25 + vector ranking
- Answer generation with source citations
- Health endpoint and FastAPI OpenAPI docs
- Test coverage for ingestion, retrieval, and API contracts

## 3. Architecture at a glance

The system is structured as a layered application:

- API layer: FastAPI endpoints in `app/main.py`
- Core configuration: environment-driven settings in `app/core/config.py`
- Domain/service logic: ingestion and question-answer orchestration in `app/services/`
- Storage adapters: original PDF persistence and vector search storage in `app/stores/`
- Deployment: containerized FastAPI service, with optional local Milvus/etcd/MinIO via Docker Compose

High-level flow:

1. A PDF is uploaded via `POST /documents`
2. The file is checked for extension and size limits
3. The PDF is parsed with PyMuPDF and text is extracted page by page
4. Text is chunked into overlapping segments
5. Each chunk is embedded using Gemini embeddings
6. Chunks and vectors are inserted into Milvus
7. User asks a question through `POST /questions`
8. The question is embedded and searched against Milvus
9. Chunks are reranked using BM25 + vector fusion
10. Gemini generates an answer grounded in the matched excerpts
11. Citations are returned as document and page references

## 4. Repository structure

```text
.
├── app/
│   ├── core/
│   │   └── config.py                 # Settings and environment configuration
│   ├── services/
│   │   ├── pdf_ingestion.py          # PDF validation, extraction, and chunking
│   │   ├── rag.py                   # Main orchestration for ingestion and QA
│   │   └── retrieval.py             # BM25 + dense fusion logic
│   ├── stores/
│   │   ├── document_storage.py      # Local document store abstraction
│   │   ├── milvus.py                # Milvus collection management and queries
│   │   └── __init__.py
│   ├── main.py                      # FastAPI app and routes
│   └── development/                 # Additional development assets or prototype code
├── data/                            # Local data directory for uploaded PDFs
├── docs/
│   └── PROJECT_DOCUMENTATION.md     # Project documentation
├── tests/
│   ├── test_api.py                  # FastAPI contract tests
│   ├── test_pdf_ingestion.py       # Chunking and PDF parsing tests
│   └── test_retrieval.py           # Retrieval ranking tests
├── .env.example                     # Sample environment variables
├── .gitignore
├── Dockerfile                       # FastAPI container image
├── .dockerignore                    # Excludes local secrets and development files from builds
├── docker-compose.yml               # FastAPI service and optional local Milvus stack
├── pyproject.toml                   # Python project and dependency config
├── README.md                        # Short project entry point
└── opencode.jsonc                   # Project-specific editor/runtime config
```

## 5. Core configuration

The application settings are defined in `app/core/config.py` and loaded from environment variables plus `.env`.

Key settings:

- `GEMINI_API_KEY`: API key for Google GenAI
- `GEMINI_EMBEDDING_MODEL`: embedding model name, default `gemini-embedding-001`
- `GEMINI_GENERATION_MODEL`: generation model, default `gemini-2.5-flash`
- `EMBEDDING_BATCH_SIZE`: texts per embedding request, default `100` (maximum `100`)
- `EMBEDDING_REQUESTS_PER_MINUTE`: client-side request pacing target, default `30` (maximum `99`)
- `EMBEDDING_RETRY_ATTEMPTS`: additional retries for Gemini 429 responses, default `3` (maximum `5`)
- `EMBEDDING_DIMENSION`: embedding vector size, default `768`
- `MILVUS_URI`: Milvus/Zilliz endpoint; local default is `http://localhost:19530`
- `MILVUS_TOKEN`: Zilliz Cloud cluster token; leave empty for unauthenticated local Milvus
- `MILVUS_COLLECTION`: vector collection name, default `rag_chunks_v2`
- `DATA_DIR`: local storage root, default `data`
- `DOCUMENT_STORAGE_BACKEND`: `local` or `r2`, default `local`
- `R2_ACCOUNT_ID`: Cloudflare account ID used to form the R2 S3-compatible endpoint
- `R2_BUCKET`: R2 bucket name for original uploaded PDFs
- `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`: R2 S3 API credentials
- `R2_PREFIX`: object key prefix, default `documents`
- `CHUNKING_STRATEGY`: `semantic` or `fixed`, default `semantic`
- `CHUNK_SIZE`: maximum chunk length, default `1000`
- `CHUNK_OVERLAP`: overlap between chunks, default `200`
- `SEMANTIC_BREAKPOINT_PERCENTILE`: distance percentile for semantic split points, default `90`
- `RETRIEVAL_CANDIDATES`: candidate count for searching, default `20`
- `TOP_K`: top results returned to the LLM, default `5`
- `RRF_K`: reciprocal rank fusion constant, default `60`
- `MAX_UPLOAD_BYTES`: file size limit, default `20 MiB`

The environment file is `.env`, copied from `.env.example` during local setup.

## 6. Runtime dependencies

The project depends on:

- FastAPI for the REST API
- Uvicorn for ASGI serving
- Boto3 for the Cloudflare R2 S3-compatible API
- PyMuPDF (`pymupdf`) for PDF parsing
- Google GenAI SDK for Gemini embeddings and answer generation
- Milvus Python client (`pymilvus`) for vector storage and retrieval
- Rank-BM25 for lexical scoring
- python-multipart for file uploads
- Pytest and Ruff for testing and linting

## 7. Local development setup

### 7.1 Requirements

- Python 3.11+
- Docker Desktop or Docker Compose compatible environment
- A valid Google Gemini API key

### 7.2 Setup steps

```bash
cp .env.example .env
# Add your GEMINI_API_KEY value to .env

docker compose up --build -d
```

This builds the FastAPI container and the Nginx reverse proxy. The API is served through Nginx at:

- Swagger UI: `http://localhost/docs`
- Health check: `http://localhost/health`

The container receives application configuration from `.env`. `.env` is excluded from the Docker
build context and is not copied into the image. Local original PDFs are persisted in the named
`app_data` volume. View logs with `docker compose logs -f api nginx` and stop containers with
`docker compose down`.

The default setup targets the configured hosted Zilliz endpoint. To use the optional local Milvus
stack, set `MILVUS_URI=http://milvus:19530` in `.env`, then run:

```bash
docker compose --profile local-milvus up --build -d
```

The `local-milvus` profile starts Milvus with its etcd and MinIO dependencies. The containerized
API must use the Compose service hostname `milvus`, rather than `localhost`, to reach it.

For non-containerized development, use `.venv/bin/pip install -e '.[dev]'` and
`.venv/bin/uvicorn app.main:app --reload`. When running the API on the host with local Milvus, use
`MILVUS_URI=http://localhost:19530`.

### 7.3 Running tests

```bash
.venv/bin/pytest
```

### 7.4 Branch-based CI and deployment

The GitHub Actions workflow runs Ruff and pytest on pushes to `main` and `develop` and pull
requests targeting either branch. Successful pushes deploy by branch:

- `main` connects to EC2 over SSH, fetches and checks out the pushed commit, rebuilds the Compose
  stack, and checks the Nginx health endpoint.
- `develop` uses a repository-restricted, macOS self-hosted runner labeled `rag-laptop` to rebuild
  and restart the local Compose stack. It uses Compose project name `rag` to keep the current local
  named volumes.

The production GitHub environment needs `EC2_HOST`, `EC2_USER`, `EC2_SSH_PRIVATE_KEY`,
`EC2_KNOWN_HOSTS`, and `EC2_DEPLOY_PATH` secrets. EC2 must have Docker Compose, Git, curl, a clean
repository checkout, a production `.env`, and a separate SSH key that can read the private repo.
Validate the EC2 host key before storing it in `EC2_KNOWN_HOSTS`. The GitHub-hosted runner must be
able to reach EC2 over SSH.

The laptop runner must be online and able to access Docker Desktop. Set the GitHub `development`
environment variable `RAG_DEV_ENV_FILE` to the absolute path of the laptop's local `.env`; the
workflow copies it into the checkout for Compose and removes that copy after the run. Only trusted
pushes deploy to this runner; pull requests run checks on GitHub-hosted runners and never run code
on the laptop.

The client uses `http://localhost` for the laptop stack and the public HTTPS domain for production.
This client API URL is distinct from the backend's `MILVUS_URI`, which configures vector database
connectivity. TLS and authentication must be configured separately from this deployment workflow.

## 8. Data flow and processing details

### 8.1 PDF ingestion

The ingestion flow begins in `POST /documents` in `app/main.py`.

The endpoint:

- accepts a multipart file upload
- accepts an optional `semantic_chunking` boolean form field to choose semantic or fixed-size chunking for that upload; if omitted, `CHUNKING_STRATEGY` is used
- validates the filename extension
- reads raw content up to the configured upload limit
- calls `validate_pdf_upload()`
- verifies the PDF contains extractable text
- generates a unique `document_id`
- saves the original PDF using the configured local or R2 document storage backend
- calls `RagService.ingest()`

The PDF validation and extraction logic is implemented in `app/services/pdf_ingestion.py`.

### 8.2 PDF extraction

`extract_pdf(content: bytes)`:

- verifies the file header starts with `%PDF-`
- opens the PDF using `pymupdf`
- rejects password-protected PDFs
- enumerates pages with 1-based page numbers
- extracts text using `page.get_text("text")`
- returns a list of `(page_number, text)` tuples

This is intentionally strict: only readable, non-password-protected PDFs are accepted.

### 8.3 Chunking

By default, ingestion uses semantic chunking. Text is split into sentences, each sentence is
embedded using the configured Gemini embedding model, and cosine distances between adjacent
sentence embeddings are used to find topic changes. Distances at or above the configured
`SEMANTIC_BREAKPOINT_PERCENTILE` become candidate boundaries. The chunk size remains a hard cap,
and adjacent chunks retain up to `CHUNK_OVERLAP` characters of context.

Set `CHUNKING_STRATEGY=fixed` to use fixed-size chunking by default, or pass
`semantic_chunking=true` or `semantic_chunking=false` as a multipart field to `POST /documents` to
override the default for one upload. Fixed-size chunking divides text into overlapping character
windows. Semantic chunking makes additional sentence-level embedding requests during ingestion, so
its latency and API usage are higher; fixed-size chunks still need embeddings for vector search.
Semantic sentences and final chunks are sent in batches of up to `EMBEDDING_BATCH_SIZE` texts per request.
Embedding calls are spaced according to `EMBEDDING_REQUESTS_PER_MINUTE` (30 requests/minute by
default, leaving room for other Gemini API traffic sharing the same project quota). When Gemini
returns a 429 response, the client honors its retry delay when available and retries up to
`EMBEDDING_RETRY_ATTEMPTS` times. If the quota is still exhausted, the API responds with HTTP 429
rather than a generic 503; check quota reset, project limits, and billing. The in-process limiter
does not coordinate across multiple application instances sharing the same API key.

Important behavior:

- `CHUNK_SIZE` must be positive and `CHUNK_OVERLAP` must be non-negative and smaller than it
- semantic boundaries are estimated independently for each PDF page; chunks never cross pages
- the sentence distance threshold is calculated from that page's adjacent-sentence distances
- fixed-size chunk boundaries are computed by stepping over `size - overlap`
- whitespace is trimmed in fixed-size chunks
- each chunk retains page-level context in the later metadata creation step

The chunk metadata includes:

- `id`: deterministic UUID based on document, page, and chunk index
- `document_id`
- `filename`
- `page`
- `chunk_index`
- `text`

This metadata is preserved when the chunks are stored in Milvus.

### 8.4 Embedding and indexing

`RagService.ingest()`:

- calls `build_chunks()` to build chunk dictionaries
- embeds the batch using `self._embed(..., "RETRIEVAL_DOCUMENT")`
- inserts both chunk metadata and vector payloads into Milvus through `MilvusStore.insert()`

`MilvusStore.ensure_collection()` ensures a Milvus collection exists and is loaded.

The collection is configured with:

- string primary field: `id`
- vector field: `vector`
- metric: cosine similarity
- dynamic fields enabled for flexible metadata storage

### 8.5 Retrieval

The retrieval pipeline in `app/services/retrieval.py`:

- tokenizes text with regex-based word matching (case-insensitive)
- builds a BM25 index over all stored chunks
- scores the query against the corpus
- takes the strongest lexical matches as candidate results
- combines dense vector results and lexical results using reciprocal rank fusion
- reorders the combined results using the lexical score as a tie-breaker and keeps the highest-ranked final set

The `hybrid_search()` function has the following significance:

- `dense_results`: vector search results from Milvus
- `chunks`: all chunks for the corpus
- `candidates`: number of results to consider per retriever
- `top_k`: number of results finally passed to the generation model
- `rrf_k`: RRF constant that controls how aggressively lower ranks are penalized

### 8.6 Answer generation

The answer step in `RagService.answer()` does the following:

- embeds the incoming question using `RETRIEVAL_QUERY`
- performs dense Milvus search
- retrieves all chunks from Milvus
- runs `hybrid_search()` to select relevant excerpts
- if no results match, returns a simple fallback answer
- otherwise builds a prompt with the retrieved JSON snippets
- calls Gemini `generate_content()`
- returns the model answer plus citations

Citation generation is deduplicated by `(document_id, page)` to keep output stable and compact.
The same retrieval pipeline is used by `RagService.stream_answer()`, which calls Gemini's
`generate_content_stream()` method and yields answer text incrementally before yielding citations.

## 9. API contract

### 9.1 `GET /health`

Returns application health status.

Example response:

```json
{
  "status": "ok"
}
```

### 9.2 `POST /documents`

Uploads a PDF file and ingests it into the PDF RAG system.

Request:

- multipart form-data
- field name: `file`
- value: PDF binary

Response model:

```json
{
  "document_id": "<uuid>",
  "filename": "example.pdf",
  "chunks": 12
}
```

Behavior:

- rejects non-PDF files
- rejects files above the configured `MAX_UPLOAD_BYTES`
- rejects password-protected or malformed PDFs
- rejects PDFs with no extractable text

### 9.3 `POST /questions`

Asks a question against the already indexed corpus.

Request body:

```json
{
  "question": "What is the main conclusion of this document?"
}
```

Response model:

```json
{
  "answer": "...",
  "citations": [
    {
      "document_id": "<uuid>",
      "filename": "example.pdf",
      "page": 3
    }
  ]
}
```

The question endpoint enforces length constraints in the request model:

- `question` is required
- minimum length: 1
- maximum length: 4000

### 9.4 `POST /questions/stream`

Accepts the same JSON body and optional `document_id` as `POST /questions`, and responds as
Server-Sent Events (`text/event-stream`). Events are emitted in this order:

```text
event: answer_chunk
data: {"text":"partial answer"}

event: citations
data: {"citations":[{"document_id":"...","filename":"example.pdf","page":3}]}
```

If a failure occurs after streaming begins, the endpoint emits an `error` event. Use
`POST /questions` when a standard JSON response and HTTP error status are preferred.

## 10. Data storage and persistence

### 10.1 Original documents

By default, original uploaded PDFs are saved locally to:

```text
data/documents/
```

Set `DOCUMENT_STORAGE_BACKEND=r2` to save original PDFs to Cloudflare R2 through its S3-compatible API. Configure `R2_ACCOUNT_ID`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`, and `R2_SECRET_ACCESS_KEY`; `R2_PREFIX` optionally sets the object-key prefix and defaults to `documents`. An upload with document ID `abc123` is stored under `documents/abc123.pdf` by default. The R2 client uses the account endpoint, `auto` region, and AWS Signature Version 4. Use a bucket-scoped R2 API token and keep its secret out of source control.

The storage adapter exposes the same `save()` interface for local and R2 backends. Selecting R2 with incomplete configuration fails the upload explicitly. Original PDF storage and vector storage are separate: only the PDF binary goes to R2, while chunks and embeddings continue to be stored in Milvus.

### 10.2 Vector storage

Milvus stores the chunk vectors and metadata. This keeps the project decoupled from application-local in-memory state and supports later scaling toward an externally managed retrieval layer.

The project README explicitly notes that the API does not keep a durable in-memory corpus and that Milvus may still need a separately hosted endpoint in production or serverless deployments.

## 11. Infrastructure and deployment notes

The project includes Docker Compose for an optional local Milvus stack:

- `etcd`
- `minio`
- `milvus`

This stack is enough for local development and experimentation. When using hosted Zilliz Cloud,
set `MILVUS_URI` to the cluster's public endpoint (including `https://`), `MILVUS_TOKEN` to its
cluster token, and `MILVUS_COLLECTION=rag_chunks_v2`. The application passes the token to the
Milvus SDK for authenticated TLS access; do not include the token in source control. Docker Compose
is not needed for this hosted setup.

The application collection requires a string primary key named `id` and a vector field named
`vector`; chunk metadata (`document_id`, `filename`, `page`, `chunk_index`, and `text`) is stored as
dynamic fields. At first use, the API creates and loads the configured collection. If an existing
collection lacks the required fields, the API reports a clear server-side error and does not alter
that collection. Use a fresh collection name such as `rag_chunks_v2`, restart the API, then upload
the PDFs again. Existing legacy collections are preserved; their stored rows are not automatically
migrated.

Important deployment considerations:

- the service expects a valid Gemini API key at runtime
- the service expects a reachable Milvus/Zilliz endpoint and valid credentials when authentication is enabled
- local file storage is not a production-grade durable solution for all workloads; R2 can be selected for persistent original-file storage
- R2 credentials must be provisioned separately and scoped to the bucket
- the project is designed to support cloud deployment with R2-backed documents and remote Milvus hosting

If `POST /questions` returns 503, inspect the API process or container logs for the recorded
exception traceback. The HTTP response intentionally stays generic. Confirm that
`MILVUS_COLLECTION` names the application-compatible collection, restart the API after changing
environment configuration, and re-upload documents if switching to a new collection.

## 12. Testing strategy

The repository includes tests for the main components:

### `tests/test_pdf_ingestion.py`

Validates:

- chunk size and overlap behavior
- blank-text handling
- invalid size/overlap rejection
- metadata preservation for document and page IDs
- upload validation rules
- PDF extraction correctness

### `tests/test_retrieval.py`

Validates:

- RRF promotes items ranked by both retrievers
- BM25 reranking limits the final top-k results
- the hybrid search result count is capped correctly

### `tests/test_api.py`

Validates:

- stable citation schema in the question API
- document upload API successfully stores the original PDF locally
- R2 object key, body, and content type are set correctly
- incomplete R2 credentials are rejected
- the API contract remains stable while dependency injection is mocked

## 13. Notable implementation decisions

### 13.1 Why hybrid retrieval?

Dense vector search is strong for semantic similarity, but lexical scoring remains important for exact term matching and keyword-heavy PDFs. Using BM25 + vector search together improves robustness when a user asks a fact-heavy question or searches for precise terminology.

### 13.2 Why citations?

The answer endpoint returns page-level citations, which is useful for trust and traceability. It lets the application point users back to the exact source pages that informed the answer.

### 13.3 Why interchangeable document storage?

The storage interface keeps upload handling independent of the storage provider. Local storage remains convenient for development, while the R2 adapter stores originals remotely without changing PDF parsing or indexing.

## 14. Limitations and current gaps

This is a useful MVP, but it still has several practical gaps:

- No multi-document collection awareness beyond storing separate document IDs
- No persistent user/session tracking
- No authentication or authorization layer
- No background job queue for heavy indexing tasks
- No PDF metadata indexing beyond textual pages
- No production-grade observability or logging
- No model versioning or A/B evaluation harness
- No built-in rate limiting or access controls

## 15. Suggested next steps

Potential future enhancements:

1. Add a proper database model for documents, users, and ingestion jobs
2. Support S3-compatible storage for uploaded PDF originals
3. Add collection-level multi-document search and filtering
4. Add document deletion and re-indexing operations
5. Improve chunking strategies for tables, headers, and references
6. Add monitoring, tracing, and structured logs
7. Introduce authentication and API quotas
8. Add a front-end or dashboard for document management and query history
9. Add evaluation tests for answer quality and citation correctness

## 16. Summary

This project is a complete, working MVP for a PDF RAG pipeline using FastAPI, Gemini, and Milvus. It covers the central workflow of ingesting PDFs, transforming them into searchable chunks, retrieving relevant excerpts, and generating grounded question answers with citations.

The codebase is intentionally organized into clear layers to make extension easier, and the current implementation is a strong base for future productionization or expansion into a larger document intelligence platform.
