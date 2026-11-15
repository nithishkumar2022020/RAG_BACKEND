# PDF RAG

A FastAPI service for PDF ingestion and cited question answering, built with Gemini, Milvus/Zilliz, and hybrid vector + BM25 retrieval.

## Run locally

```sh
cp .env.example .env
# Add the required Gemini and Milvus settings to .env
docker compose up --build -d
```

The API is available through Nginx at `http://localhost`; open `http://localhost/docs` for the interactive API. Configure secrets and storage in `.env` (never commit it).

## Key features

- Asynchronous PDF ingestion with job-status polling.
- Configurable semantic or fixed-size chunking.
- Cited answers, including a streaming endpoint.
- Milvus/Zilliz vector search with BM25 retrieval.
- Optional Cloudflare R2 storage for original PDFs.

## Checks

```sh
.venv/bin/pytest
.venv/bin/ruff check .
python -m app.evaluation --mode offline
```

## Documentation

The detailed, searchable project guide—including setup, architecture, API, operations, evaluation dataset and metrics, deployment, and learning roadmap—is available on the [documentation website](https://nithishkumar2022020.github.io/RAG_BACKEND/).

The source documentation is in [`docs/`](docs/). To preview it locally:

```sh
python -m pip install -e '.[docs]'
python -m mkdocs serve
```

## CI and publishing

GitHub Actions runs lint, tests, and offline evaluation on pushes and pull requests to `main` and `develop`. A push to `main` publishes the container image to GHCR; a push to `develop` deploys through the configured self-hosted runner. See the [deployment guide](https://nithishkumar2022020.github.io/RAG_BACKEND/PROJECT_DOCUMENTATION/) for setup details.

## License note

PyMuPDF is AGPL-3.0 or commercially licensed. Review its terms before distributing a closed-source derivative.
