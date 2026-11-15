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

The `POST /documents` multipart form accepts an optional `semantic_chunking` boolean alongside `file`. Send `true` to use semantic chunking for that upload or `false` to use fixed-size chunking. If omitted, the server uses `CHUNKING_STRATEGY` from its environment. Upload now returns `202 Accepted` with a `job_id` immediately; poll `GET /documents/{job_id}` until its status is `completed` or `failed`, rather than holding the upload request open while Gemini processes a large PDF:

```sh
curl -F 'file=@document.pdf' -F 'semantic_chunking=false' http://localhost/documents
curl http://localhost/documents/<job_id>
```

The job status response includes the generated `document_id`; use it to query the document after ingestion completes. Turning semantic chunking off skips the extra per-sentence embedding calls, which can reduce Gemini quota usage and ingestion time. The resulting fixed-size chunks still need embeddings for vector search, so this does not remove all Gemini embedding requests. Job status is held in the API process memory and is lost if the process/container restarts; don't restart the API while a large ingestion is running.

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
boundaries. Set `CHUNKING_STRATEGY=fixed` in `.env` to use fixed-size chunking by default; the
per-upload `semantic_chunking` form field can override this default. Embedding inputs are sent in
batches of up to 100 and scheduled at 30 requests per minute by default. Both values are configurable
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

A theory-first list of concepts to learn in order to understand and maintain this codebase is available in [docs/LEARNING_ROADMAP.md](docs/LEARNING_ROADMAP.md). This is the project's running learning list and will be extended as new concepts are introduced.

### RAG evaluation

The checked-in fictional golden dataset can be evaluated offline without Gemini credentials, Milvus, network access, or real user documents:

```sh
python -m app.evaluation --mode offline
```

By default, the JSON report is written to the ignored `eval-results/latest.json`. Compare with the checked-in initial baseline (same dataset and `--top-k` cutoff required):

```sh
python -m app.evaluation --mode offline --output eval-results/current.json \
  --compare evaluation/baseline.json
```

Offline metrics include retrieval Hit Rate/Recall@K, MRR, nDCG@K, and citation precision/recall. The live mode uses Gemini and Milvus, creates a temporary uniquely named `rag_eval_*` collection, indexes only the fictional fixture corpus, evaluates answers, then drops that collection:

```sh
python -m app.evaluation --mode live --output eval-results/live.json
python -m app.evaluation --mode live --judge --output eval-results/live-judged.json
```

Live mode requires the configured Gemini and Milvus credentials and may consume API quota. The optional `--judge` makes additional Gemini calls. Reports include answer token precision/recall/F1 and stage timings; a judge can add faithfulness/relevance scores. Never put real documents, secrets, or sensitive outputs in the checked-in golden dataset or commit generated reports. See the [evaluation guide](docs/EVALUATION.md) for metric definitions and how to extend the dataset.

### Documentation website

The Markdown docs are presented as a searchable, responsive site using MkDocs Material and GitHub Pages. After enabling Pages with **GitHub Actions** as the source, a successful push to `develop` publishes the site at <https://nithishkumar2022020.github.io/RAG_BACKEND/>. The docs build is checked on both branches, but `develop` is the canonical publishing branch so an older `main` commit cannot overwrite newer docs. Pages stays available between pushes; each successful `develop` push republishes the latest docs.

To preview the docs locally:

```sh
python -m pip install -e '.[docs]'
python -m mkdocs serve
```

Open the local URL printed by MkDocs (normally `http://127.0.0.1:8000`). The generated `site/` directory is a build artifact and should not be edited or committed.

## Continuous integration

GitHub Actions runs Ruff and pytest on pushes to `main` and `develop`, and on pull requests targeting either branch. Pull requests only run checks; deployment jobs run only after successful checks on branch pushes:

- A push to `main` deploys the checked commit to EC2 over SSH and rebuilds the Docker Compose stack.
- A push to `develop` deploys the checked commit to this laptop through a self-hosted GitHub Actions runner.

### Configure the EC2 deployment

Create a GitHub Actions environment named `production` and add these environment secrets:

- `EC2_HOST`: the EC2 public DNS name or IP reachable from GitHub-hosted runners.
- `EC2_USER`: a dedicated EC2 deployment user.
- `EC2_SSH_PRIVATE_KEY`: the private key for that deployment user.
- `EC2_KNOWN_HOSTS`: the verified SSH host-key line for the EC2 host; do not trust an unverified key scan.
- `EC2_DEPLOY_PATH`: an absolute path to a clean clone of this repository on EC2.

On EC2, install Docker Engine, the Docker Compose plugin, Git, and curl. Set up the deployment user's SSH key to read this private GitHub repository, clone it into `EC2_DEPLOY_PATH`, and create the production `.env` there with the required Gemini, Milvus/Zilliz, and optional R2 settings. The workflow fetches the pushed `main` commit, checks it out, runs `docker compose up -d --build --remove-orphans`, and checks `http://127.0.0.1/health`. Keep `.env` out of Git and preserve the Compose volumes. Ensure EC2's network rules allow SSH from the chosen GitHub Actions runner and expose only the required web ports publicly. For a public service, configure HTTPS at a trusted load balancer or TLS-enabled proxy and use that HTTPS domain in the client.

### Configure the laptop `develop` deployment

Install a GitHub self-hosted runner on the laptop, restricted to this repository, and give it the custom label `rag-laptop` (the workflow also requires the `macOS` label). Run the runner as the same user that can access Docker Desktop. Keep the laptop awake and the runner online for deployments. In a GitHub Actions environment named `development`, set the variable `RAG_DEV_ENV_FILE` to the absolute path of the laptop's local `.env` file. The runner copies this file into its temporary checkout for Compose and removes the copy after deployment; the source `.env` is not committed.

The local job uses Compose project name `rag` to reuse this project's existing named volumes, rebuilds the stack, and checks `http://localhost/health`. The local client should use `http://localhost` through Nginx; the production client should use the EC2 HTTPS domain. Configure those client base URLs in the client/environment configuration—the backend deployment workflow does not rewrite a separate Flutter app's API URL. The backend can still use its own configured Milvus and Gemini endpoints; the client's `localhost` URL is not the Milvus address.

Only trusted pushes to `develop` execute code on the laptop runner; never run untrusted pull-request code there. Protect `main` and `develop` with appropriate branch rules, and require **Python checks** for merges. The `develop` branch must exist on GitHub before its deployment job can run. Deployments do not add API authentication or TLS by themselves.
