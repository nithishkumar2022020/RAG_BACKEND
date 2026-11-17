# PDF RAG

A practical guide to the PDF question-answering service: how it works, how to run it,
and what to learn to understand and extend it.

<div class="grid cards" markdown>

-   :material-file-document-multiple: **Project guide**

    ---

    Follow a PDF from upload through extraction, chunking, retrieval, and cited answers.

    [:octicons-arrow-right-24: Read the project documentation](PROJECT_DOCUMENTATION.md)

-   :material-school: **Learning roadmap**

    ---

    Explore the concepts behind the project in a suggested theory-first order.

    [:octicons-arrow-right-24: Open the learning checklist](LEARNING_ROADMAP.md)

-   :material-chart-box-outline: **RAG evaluation**

    ---

    Compare retrieval and answer quality with offline metrics and optional live runs.

    [:octicons-arrow-right-24: Run the evaluation harness](EVALUATION.md)

</div>

## What this project does

The FastAPI backend accepts PDF uploads, extracts page-level text, creates chunks and Gemini
embeddings, stores searchable data in Milvus/Zilliz, and answers questions with citations.
Retrieval combines vector similarity with BM25 text matching. Nginx serves as the HTTP
reverse proxy and supports streaming answers.

## Run locally

The easiest way to run the full service is with Docker Compose:

```sh
cp .env.example .env
# Add the required Gemini and Milvus settings to .env
docker compose up --build -d
```

The local API is available at `http://localhost` through Nginx. Use `http://localhost/docs`
for the interactive API reference. See the [project guide](PROJECT_DOCUMENTATION.md) for
configuration, local Milvus, storage options, and troubleshooting.

## Where to start

1. Read the [architecture and data flow](PROJECT_DOCUMENTATION.md).
2. Use the [learning roadmap](LEARNING_ROADMAP.md) to review underlying concepts.
3. Try the setup and API requests in the project guide.
4. Follow the relevant service and test code as you study each topic.

!!! note "Documentation source"

    These pages are generated from Markdown committed with the project. Edit the Markdown
    files—not the generated site output—when updating documentation.
