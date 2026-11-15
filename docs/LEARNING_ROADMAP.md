# Project Learning Roadmap

This is the running list of concepts behind this project. It is intended to help you
understand what the code is doing and why, even if you are learning the theory before
making changes yourself.

**Keep this list current:** when future work adds a technology, design pattern, or
operational concern, add its name and a short explanation under the relevant section.
Add a new section when a concept does not fit an existing one.

## Suggested learning order

Learn the sections in order. The first sections explain the application and RAG
pipeline; the later sections cover deploying and operating it.

## 1. Python and project structure

- [ ] **Python modules and packages** — How code is split across `app/`, and how imports connect the API, services, and storage layers.
- [ ] **Functions, classes, and dependency injection** — How `RagService` and storage classes receive the objects they need instead of constructing every dependency inside each function.
- [ ] **Type hints and collection types** — Annotations such as `list[dict]`, optional values, and iterators; they communicate expected inputs and outputs.
- [ ] **Protocols and adapters** — A `Protocol` describes an interface, while local disk and R2 implementations provide different ways to satisfy it.
- [ ] **Exceptions and error propagation** — How failures move from SDKs and stores to service code and finally become HTTP errors.
- [ ] **Logging** — Recording useful operational context and tracebacks without returning internal details to API callers.
- [ ] **Virtual environments and package installation** — Isolating dependencies and installing this project from `pyproject.toml`.
- [ ] **Python packaging and `pyproject.toml`** — Runtime dependencies, development dependencies, package discovery, and test/lint configuration.
- [ ] **Environment-based configuration** — How Pydantic Settings reads environment variables and `.env`, validates values, and supplies defaults.
- [ ] **Caching configuration** — What `lru_cache` does when settings or service factories are cached.

## 2. HTTP APIs and FastAPI

- [ ] **HTTP fundamentals** — Requests, responses, headers, body formats, methods, status codes, and content types.
- [ ] **REST API design** — Why document upload and question answering are represented as separate endpoints.
- [ ] **FastAPI routes and dependency injection** — How endpoint functions, `Depends`, and application objects are connected.
- [ ] **Pydantic request and response models** — Input validation, required and optional fields, and stable response shapes.
- [ ] **Multipart file uploads** — How a PDF is sent as `multipart/form-data` and how FastAPI receives it as an `UploadFile`.
- [ ] **ASGI and Uvicorn** — The asynchronous server interface and the process that serves the FastAPI application.
- [ ] **OpenAPI and Swagger UI** — The automatically generated API schema and interactive `/docs` page.
- [ ] **Health checks and readiness** — A basic process health endpoint versus checks that confirm external dependencies are reachable.
- [ ] **HTTP error mapping** — Returning appropriate errors such as 422 for invalid input, 429 for exhausted quotas, and 503 for service failures.

## 3. PDF ingestion and text processing

- [ ] **Bytes, file signatures, and upload validation** — Why checking a `.pdf` filename alone is not enough and how size limits help protect the service.
- [ ] **PDF parsing with PyMuPDF** — Opening PDF bytes, iterating pages, extracting text, and handling unreadable or encrypted documents.
- [ ] **Page-level metadata** — Keeping page numbers with extracted text so answers can cite their source.
- [ ] **Text normalization and sentence splitting** — Preparing extracted text for chunking while recognizing that PDF text may have awkward line breaks.
- [ ] **OCR versus text extraction** — Why scanned/image-only PDFs need optical character recognition, which this project's current extraction path does not provide.
- [ ] **Deterministic identifiers** — Creating repeatable chunk IDs from document, page, and chunk position.

## 4. RAG and language-model concepts

- [ ] **Retrieval-Augmented Generation (RAG)** — Finding relevant source text first, then giving it to a language model to ground an answer.
- [ ] **Tokens and context windows** — Why prompt size matters, why this service caps whole-document context, and why character count is only an approximation of token count.
- [ ] **Prompt construction** — Separating the question from JSON-encoded excerpts and giving the model instructions for how to use them.
- [ ] **System instructions and API versions** — How model behavior instructions are sent separately from user content, and why the Gemini API version must support that feature.
- [ ] **Grounding, hallucinations, and citations** — Asking the model to support claims with retrieved excerpts and returning document/page citations.
- [ ] **Safety boundaries for retrieved content** — Treating document text as data rather than instructions to the model.
- [ ] **Judgment and summary questions** — Why evaluation prompts need relevant coverage from the selected document rather than only a few top vector matches.

## 5. Embeddings, chunking, and retrieval

- [ ] **Text embeddings** — Representing text as numeric vectors so semantically similar text can be compared.
- [ ] **Embedding task types and dimensions** — Why the project labels embedding inputs as documents or queries and configures a vector dimension.
- [ ] **Fixed-size chunking and overlap** — Splitting text into bounded windows while repeating some text across boundaries to preserve context.
- [ ] **Semantic chunking** — Grouping sentences using embedding similarity so topic changes can become chunk boundaries.
- [ ] **Per-upload feature configuration** — How an optional multipart form field can override the server's default chunking strategy for one document upload.
- [ ] **Cosine similarity and distance** — Measuring the angle between vectors and using a change in similarity as a signal for a topic boundary.
- [ ] **Percentiles and semantic breakpoints** — Using a distance percentile to decide which changes are large enough to split.
- [ ] **Chunk-size tradeoffs** — How chunk length and overlap affect retrieval precision, context, model cost, and ingestion time.
- [ ] **Vector databases and similarity search** — How Milvus stores vectors and returns nearest matches for a query vector.
- [ ] **Milvus collections and schemas** — Collections, primary keys, vector fields, metadata fields, dimensions, and why incompatible schemas require a new collection or migration.
- [ ] **Metadata filters** — Restricting searches to a selected document or set of documents.
- [ ] **Dense retrieval** — Finding semantically related chunks with vector similarity.
- [ ] **Lexical retrieval and BM25** — Finding chunks with matching terms using a text-based relevance score.
- [ ] **Hybrid retrieval** — Combining vector and lexical results to cover both semantic matches and exact terms.
- [ ] **Reciprocal Rank Fusion (RRF)** — Combining ranked lists without requiring their scores to use the same scale.
- [ ] **Candidates and top-K** — The difference between retrieval candidates and the smaller number of excerpts sent to the model.
- [ ] **Whole-document retrieval and fallback** — Selecting all chunks for summary/evaluation questions, then falling back to top matches when context is too large.
- [ ] **Retrieval evaluation** — Checking whether the relevant source appears in retrieved results, separately from whether the model writes a good answer.

## 6. Gemini API behavior and resilience

- [ ] **Generation versus embedding API calls** — Different model operations used for answer generation and vector creation.
- [ ] **Streaming generation** — Consuming model output incrementally rather than waiting for a complete response.
- [ ] **Batching** — Sending multiple embedding texts per API request to reduce call count.
- [ ] **Rate limits and quotas** — Provider request limits, how they differ from local pacing, and why multiple app instances can exceed a shared quota.
- [ ] **Retries and backoff** — Retrying transient quota failures, respecting provider retry hints, and limiting retries to avoid retry storms.
- [ ] **Provider errors versus application errors** — Preserving useful diagnostics in logs while exposing safe, understandable responses to clients.
- [ ] **API credentials** — Keeping Gemini keys and other secrets in runtime configuration rather than source code, images, or logs.

## 7. Streaming and client integration

- [ ] **Server-Sent Events (SSE)** — The `text/event-stream` response format, `event:` and `data:` fields, and the blank line that ends each event.
- [ ] **POST-based streaming clients** — Why this endpoint uses a JSON POST and why native browser `EventSource` alone is not suitable for that request shape.
- [ ] **Incremental UI updates** — Reading partial answer events, appending text, and handling final citations and error events on the client.
- [ ] **Streaming error timing** — Why an error after response headers have been sent must be represented as a stream event instead of a new HTTP status.
- [ ] **Background jobs and polling** — Returning an accepted job ID quickly, processing long-running work outside the request, and polling a status endpoint until completion.
- [ ] **In-memory queue limitations** — Worker concurrency, queue backpressure, process-local job state, and why durable jobs need a database or external queue.
- [ ] **Flutter client basics** — Where a Flutter app fits as an API consumer and the need for an HTTP client capable of reading a POST response stream.

## 8. Storage and persistence

- [ ] **Storage abstraction** — Keeping service code independent from the specific system used to save original PDFs.
- [ ] **Local filesystem storage** — Writing files to a directory and the implications of using instance-local storage.
- [ ] **Object storage concepts** — Buckets, object keys, prefixes, credentials, and durable storage outside an application container.
- [ ] **Amazon S3 compatibility and Cloudflare R2** — Using Boto3's S3 API against R2's compatible endpoint.
- [ ] **Vector storage versus document storage** — The project stores original PDF bytes separately from extracted chunks and their vectors.
- [ ] **Persistence, backup, and recovery** — Why a Docker volume is not itself a backup and how to plan for instance or volume loss.

## 9. Docker and Docker Compose

- [ ] **Container images and containers** — The difference between the build artifact and a running instance of it.
- [ ] **Dockerfile layers and build context** — How the application image is built and why `.dockerignore` excludes local files and secrets.
- [ ] **Container networking and service DNS** — How Compose services reach each other by service name, and why `localhost` inside a container means that same container.
- [ ] **Published ports** — The difference between a container-only port and a port exposed on the host.
- [ ] **Volumes** — Persisting uploaded documents and database state beyond an individual container lifecycle.
- [ ] **Compose profiles** — Keeping the optional local Milvus, etcd, and MinIO services separate from the default hosted-database setup.
- [ ] **Health checks and startup dependencies** — Waiting for a service to be healthy before starting its dependent proxy.
- [ ] **Restart policies and logs** — Restart behavior and the use of `docker compose logs` to diagnose containers.
- [ ] **Runtime configuration and secrets** — Passing environment values to containers without baking secrets into images.

## 10. Nginx reverse proxy

- [ ] **Reverse proxy fundamentals** — Receiving public HTTP requests and forwarding them to a private upstream API.
- [ ] **Proxy headers** — Forwarding the original host, client address, and scheme to the application.
- [ ] **SSE proxy buffering** — Disabling buffering so streaming chunks reach the client promptly.
- [ ] **Proxy timeouts** — Allowing long-running document processing and model generation without premature disconnects.
- [ ] **Request body limits** — Matching Nginx's upload allowance to the API's configured PDF size limit.
- [ ] **Container DNS resolution** — Resolving an upstream by Docker service name as containers restart or change address.
- [ ] **TLS termination** — Where HTTPS certificates are configured, such as Nginx or an AWS load balancer; the current Compose Nginx setup is HTTP-only.

## 11. Testing, CI, and Git

- [ ] **Golden datasets and evaluation cases** — Curated questions, references, graded relevant context, and citations used as repeatable quality tests.
- [ ] **Information retrieval metrics** — Hit Rate/Recall@K, Mean Reciprocal Rank, and normalized Discounted Cumulative Gain for ranking results.
- [ ] **Answer token precision/recall/F1** — A simple overlap metric for comparing generated answers with reference answers, including its limitations.
- [ ] **Citation precision and recall** — Measuring whether cited document pages match the curated supporting pages.
- [ ] **LLM-as-judge evaluation** — Using a model to estimate faithfulness/relevance, with added cost, prompt sensitivity, and score variability.
- [ ] **Evaluation baselines and reports** — Saving per-case and aggregate metrics with code/config metadata to compare changes across runs.
- [ ] **Live versus offline evaluation** — Separating deterministic no-network regression tests from provider-backed end-to-end runs.
- [ ] **Unit and API tests with pytest** — Testing chunking, retrieval, endpoint behavior, and response contracts.
- [ ] **Mocks and fakes** — Replacing Gemini, Milvus, or service objects in tests so tests do not need real cloud credentials.
- [ ] **Test doubles and interface drift** — Keeping fakes aligned with production method parameters when the service interface changes.
- [ ] **Linting with Ruff** — Finding style and common code issues automatically.
- [ ] **Continuous Integration (CI)** — Running checks on proposed changes through GitHub Actions.
- [ ] **Docs-as-code and static site generation** — Keeping documentation in versioned Markdown and generating a navigable HTML site with MkDocs Material.
- [ ] **GitHub Pages publishing** — Building a static site artifact in CI and publishing it with GitHub Actions permissions and environments.
- [ ] **Workflow triggers and status checks** — Pull-request/push triggers, job names, and making checks required through GitHub branch protection.
- [ ] **Branch-based continuous deployment (CD)** — Using branch-specific jobs to deploy validated commits to distinct environments.
- [ ] **Self-hosted GitHub Actions runners** — Running automation on a laptop, runner labels, availability requirements, and the security risk of executing repository code on a personal machine.
- [ ] **SSH deployment and host-key verification** — Authenticating to EC2 using a scoped key and verifying the server identity rather than blindly trusting a network scan.
- [ ] **GitHub Environments and scoped secrets** — Separating production/development variables and credentials from committed workflow files.
- [ ] **Deployment concurrency and health checks** — Preventing overlapping releases and checking service readiness after a deployment.
- [ ] **Git branches, diffs, and staged changes** — Reviewing exactly what a change will include before committing or pushing.
- [ ] **Reproducible dependency setup** — Installing the same declared project and development dependencies in a clean CI environment.

## 12. AWS EC2 deployment and operations

- [ ] **EC2 instances and Linux services** — Provisioning the virtual machine that hosts Docker and the Compose services.
- [ ] **Security groups and network exposure** — Allowing only required inbound traffic; avoid exposing the API or database directly to the public internet.
- [ ] **DNS and domain names** — Pointing a domain to an EC2 address or load balancer.
- [ ] **HTTPS certificates** — Issuing, installing, and renewing TLS certificates, or terminating TLS at an AWS load balancer.
- [ ] **AWS IAM roles** — Giving an EC2 instance limited AWS permissions without embedding long-lived AWS credentials.
- [ ] **Secrets Manager and Parameter Store** — Supplying provider credentials securely at runtime.
- [ ] **Deployment and rollback** — Updating images/configuration safely and restoring the previous known-good version if checks fail.
- [ ] **Logs, metrics, and alerts** — Tracking application failures, container health, CPU, memory, disk, and provider/API availability.
- [ ] **Capacity and cost planning** — Estimating memory/CPU needs, storage growth, API quota consumption, and cloud costs.
- [ ] **Production readiness versus a working demo** — Authentication, HTTPS, backups, monitoring, recovery, and access controls are separate requirements from merely starting the containers.

## Concepts added as the project evolves

This section can record concepts introduced by future work, with a date and a short reason
for adding them. The categorized sections above are the current baseline from the project
work so far.
