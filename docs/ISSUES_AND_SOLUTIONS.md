# Issues encountered and resolutions

This is the running incident and troubleshooting log for the PDF RAG project and its
SYNQRA client integration. It records the observed symptom, cause, resolution, verification,
and any remaining follow-up so the same issue is easier to diagnose next time.

The entries combine issues investigated during development and deployment preparation. A
code change being complete does not necessarily mean a live deployment has been verified:
the **status** below distinguishes those cases.

## Status meanings

- **Resolved and verified** — The change was made and the relevant local check passed.
- **Resolved in code; deployment verification pending** — The implementation is present,
  but still needs to be exercised in the target environment.
- **Environment action required** — A GitHub, AWS, device, or provider setting must be
  changed outside this repository.
- **Known limitation** — The behavior is understood and documented, but it remains a
  design or operational constraint.

## Incident index

| # | Symptom | Status |
|---|---|---|
| 1 | Port 8000 appeared blank in the browser | Resolved and verified |
| 2 | Gemini rejected developer instructions using API version `v1` | Resolved in code; later stream failures need normal live monitoring |
| 3 | Embedding calls hit Gemini quota/rate limits | Mitigated; quotas remain provider-dependent |
| 4 | Large PDF uploads timed out | Resolved in code; in-memory queue is a known limitation |
| 5 | Ingestion/search behavior needed semantic chunking to be switchable | Resolved in code |
| 6 | Existing Zilliz collection did not match the application schema | Workaround documented; legacy data is not migrated |
| 7 | RAG quality changes needed repeatable evaluation | Resolved and verified for offline evaluation |
| 8 | README was too long for an entry point | Resolved; details remain in the docs site |
| 9 | GitHub Pages deployment was rejected or skipped | Workflow fixed; environment rule must allow `develop` |
| 10 | Laptop deploy waited behind an old run, then waited for a runner | Workflow policy fixed; requires online registered runner |
| 11 | Laptop deploy could not find the configured `.env` | Diagnostics improved; runner variable and file access must be verified |
| 12 | Milvus runtime files repeatedly appeared as Git changes | Ignore/untrack change was made; preserve local data |
| 13 | GHCR image publishing and EC2 deployment instructions changed over time | Current flow is `main` → GHCR → EC2; verify configured access |
| 14 | EC2 Nginx did not publish host port 80 | Workflow now pins Compose inputs and checks publication; live EC2 check pending |
| 15 | Public API needed authentication and user-level data separation | Implemented for Cognito; frontend/configuration and rollout steps remain |
| 16 | SYNQRA's default URL did not match the deployed Compose ports | Correct production URL/TLS and frontend migration still required |

## 1. Port 8000 appeared blank in the browser

**Observed:** Opening `http://localhost:8000` showed no page, although the server appeared
to be running.

**Cause:** FastAPI did not define a root (`/`) route. The service was reachable through
health and API routes, and the Compose deployment exposed Nginx on host port 80 rather than
publishing FastAPI port 8000 directly.

**Resolution:** Use `/health` to check service availability, `/docs` for the interactive
API when running in development, and the Compose Nginx entry point at `http://localhost`.
The container's FastAPI port remains internal to the Compose network.

**Status:** Resolved and verified at the time: `/health` and `/docs` returned 200; `/`
returned 404 as expected.

## 2. Gemini rejected developer instructions on API version `v1`

**Observed:** Streaming question answering failed with a Gemini `400 INVALID_ARGUMENT`
stating that developer instruction was not enabled for API version `v1`.

**Cause:** The generation request used a system/developer instruction with an API version
that did not support it.

**Resolution:** Configure the Google GenAI client to use `v1beta`, where the instruction
feature is supported. Later streaming errors were separately investigated; an error during
stream iteration can happen after the HTTP response has started, so the stream endpoint
reports a terminal SSE `error` event rather than changing the already-started HTTP status.

**Status:** API-version mismatch resolved in code. Provider availability, quota, and
account configuration still need live monitoring.

## 3. Gemini embedding quota and rate-limit failures

**Observed:** Ingestion or question requests failed with provider quota/rate-limit errors,
including HTTP 429 responses.

**Cause:** Chunking and retrieval generate embedding requests. Semantic chunking can make
additional embedding calls for sentences, in addition to the embedding calls required for
the final chunks and queries. A local application cannot exceed the Gemini project's
provider quota.

**Resolution and mitigations:**

- Batch embedding inputs, capped at 100 items per request.
- Pace embedding requests with configurable `EMBEDDING_BATCH_SIZE` and
  `EMBEDDING_REQUESTS_PER_MINUTE`.
- Retry provider 429 responses using provider retry hints when available, with bounded
  backoff.
- Return/report quota exhaustion explicitly rather than hiding it as a successful answer.
- Allow fixed-size chunking per upload or via `CHUNKING_STRATEGY=fixed` to avoid the extra
  sentence-embedding work of semantic boundary detection.

Disabling semantic chunking does **not** eliminate embeddings: chunks still require vectors
for dense retrieval.

**Status:** Mitigated in code. Actual quotas and billing remain controlled by the Gemini
project.

## 4. Large PDF uploads timed out

**Observed:** Large PDFs took longer to process than the upload request's HTTP timeout.

**Cause:** The request originally waited for extraction, chunking, embedding, and indexing
to finish.

**Resolution:** `POST /documents` now validates/queues the upload and responds with
`202 Accepted` and a `job_id`. The client polls `GET /documents/{job_id}` for `queued`,
`processing`, `completed`, or `failed`. The queue is deliberately bounded to one active
ingestion plus one waiting job; a full queue returns HTTP 429.

**Known limitation:** Job state and queued work are held in process memory. A restart loses
job status and may interrupt an ingestion. This is not a durable distributed job queue.

**Status:** Resolved in code and covered by tests. Production should avoid restarting the
API during ingestion; a durable queue is a future improvement if workload requires it.

## 5. Semantic chunking needed to be configurable per upload

**Observed:** The ingestion behavior needed to switch between semantic and fixed-size
chunking, including when semantic embedding work caused quota pressure.

**Cause:** A single server-wide chunking strategy was not flexible enough for individual
uploads or for comparing strategies.

**Resolution:** `POST /documents` accepts an optional `semantic_chunking` multipart boolean.
When omitted, the server uses `CHUNKING_STRATEGY`; when supplied, it selects the strategy
for that upload. The frontend stores its preference and sends the setting.

**Status:** Implemented and covered by backend tests. The client should keep sending the
field only when its user-facing setting is intended to override the backend default.

## 6. Zilliz collection schema mismatch

**Observed:** Retrieval or indexing against an existing Zilliz collection failed because
its schema did not have the fields the application expects.

**Cause:** Existing/legacy collection schemas cannot safely be assumed to match the
application's current required `id` and `vector` fields and dynamic chunk metadata.

**Resolution:** Configure a fresh collection (the documented name is `rag_chunks_v2`) and
re-ingest the PDFs. The application validates the collection schema and reports an
incompatibility rather than silently migrating or rewriting it.

**Status:** Workaround is documented and validated by tests. Legacy collections and their
data are not migrated automatically; retain them until the replacement index is verified.

## 7. RAG changes needed a repeatable evaluation harness

**Observed:** Retrieval/chunking changes needed a consistent way to compare behavior
between revisions.

**Resolution:** Added an offline deterministic evaluator and an optional live Gemini/Milvus
mode. The checked-in synthetic set contains two fictional documents, five page fixtures,
and six questions. Offline evaluation scores retrieval and citations without provider calls;
live mode adds answer overlap, timing, and an optional Gemini judge. CI compares the offline
report with a checked-in baseline.

The initial baseline at K=5, 20 candidates, and RRF K=60 was:

| Metric | Baseline |
|---|---:|
| Hit Rate@5 | 1.000 |
| Recall@5 | 1.000 |
| MRR@5 | 0.917 |
| nDCG@5 | 0.938 |
| Citation precision | 0.367 |
| Citation recall | 1.000 |
| Citation F1 | 0.528 |

This is a small synthetic starter set, not a production-quality claim. See the
[evaluation guide](EVALUATION.md) for dataset contents, test coverage, metrics, and commands.

**Status:** Offline harness and baseline comparison verified. Live provider-backed
evaluation is opt-in and was not used for CI validation.

## 8. README became too detailed

**Observed:** The repository README had accumulated setup, operations, evaluation, and
deployment details that made it unwieldy as a project entry point.

**Resolution:** Keep the README short, with project summary, quick start, common checks,
and links to the searchable MkDocs website. Move durable detail into the project guide,
evaluation guide, learning roadmap, and this issue log.

**Status:** Resolved. The README is intentionally concise; the docs website is the detailed
source of truth.

## 9. GitHub Pages deployment was rejected or skipped

**Observed:** GitHub reported that branch `develop` was not allowed to deploy to the
`github-pages` environment. Another workflow revision had invalid branch expressions that
could cause the artifact upload/deploy jobs to be skipped.

**Cause:** Two separate gates were involved:

1. The workflow branch condition must match the intended docs branch.
2. The GitHub environment's deployment-branch allowlist must also permit that branch.

**Resolution:** The workflow conditions were corrected to run Pages artifact upload and
deployment only on pushes to `develop`. In repository settings, add `develop` to
**Settings → Environments → `github-pages` → Deployment branches and tags**. The settings
page had previously shown only `main` in the allowlist. A later Actions screenshot showed
the Pages job succeeding.

**Status:** Workflow condition fixed. Keep the environment allowlist aligned with `develop`;
environment settings are managed in GitHub, not in this repository.

## 10. Laptop deployment waited behind an obsolete run, then had no runner

**Observed:** `Deploy develop to laptop` first waited for an earlier deployment in the same
concurrency group. Once eligible, it waited for a matching runner; initially no self-hosted
runner was registered.

**Cause:** The workflow serialized deployments with `cancel-in-progress: false`; later, the
job required a runner labeled `self-hosted`, `macOS`, and `rag-laptop`, but GitHub had no
online runner with those labels.

**Resolution:** Set `cancel-in-progress: true` for the laptop deployment so a newer
`develop` run supersedes an obsolete one. A repository-scoped macOS self-hosted runner was
then created with the required labels.

**Status:** Workflow policy and runner setup were addressed. The runner must stay online,
awake, and able to use Docker Desktop. Because it executes trusted branch code on a laptop,
protect pushes to `develop` and never run untrusted pull-request code on it.

## 11. Laptop deployment could not find its `.env`

**Observed:** The deploy job failed with `Configure development variable RAG_DEV_ENV_FILE to
a readable local .env file.`

**Cause:** The workflow reads `${{ vars.RAG_DEV_ENV_FILE }}` from the `development`
environment, then checks whether the runner process can read that exact path. A path on the
developer's machine is not automatically visible to GitHub-hosted runners; this job must
run on the self-hosted Mac as the same user that can access the file. The intended path
provided was `/Users/nithishkumarreddy/Documents/secrets/rag-secrets/.env`.

**Resolution:** Configure `RAG_DEV_ENV_FILE` as an environment **variable** (not a secret)
under **Settings → Environments → `development`**, using the absolute path visible to the
runner. The workflow now emits separate diagnostics for an empty variable, non-absolute
path, missing file, or unreadable file. It copies the file into the checkout for Compose
and removes that copy afterward.

**Status:** Diagnostics improved. The file was verified readable by the interactive local
shell; that alone does not prove the runner service account can read it. Confirm on a runner
job and keep the original file outside Git.

## 12. Milvus files repeatedly appeared as Git changes

**Observed:** Etcd snapshots/WALs and Milvus database files appeared as modified files
after normal database activity.

**Cause:** `milvus/volumes/` held live database state but had already been committed to Git.
`.gitignore` does not stop Git tracking files that are already in the index.

**Resolution:** Add `/milvus/volumes/` to `.gitignore` and remove those files from Git's
index with `git rm --cached`, preserving the local directory and data.

**Important:** Do not delete the live Milvus data directory as part of this cleanup. Once
the index removal is committed, new runtime changes under this path should no longer show
as ordinary tracked-file modifications.

## 13. Container registry and production deployment flow changed

**Observed:** Earlier workflow runs referred to ECR or an EC2 SSH deploy from an older
workflow version; the intended image registry and release sequence changed.

**Resolution:** The current intended path is:

- Push to `main`: run checks, build and push the API image to GHCR tagged by commit SHA
  (and `latest`), then deploy that exact SHA image to EC2.
- Push to `develop`: run checks, deploy source on the laptop runner, and publish the docs
  website.

An `Invalid EC2_HOST` error came from an older workflow run that still validated/used an
EC2 SSH deployment configuration. Always inspect the workflow file at the commit that
actually ran; an old Actions run does not reflect later workflow changes.

**Status:** Branch-specific workflow code exists. Successful production deployment still
depends on GitHub `production` secrets, EC2 setup, GHCR read access, and an online EC2
instance.

## 14. EC2 Nginx did not publish host port 80

**Observed:** The API container was healthy, but the Nginx container showed no published
ports; the deployment's final `curl http://127.0.0.1/health` failed with connection
refused.

**Configuration evidence:** The repository Compose file declares Nginx with `80:80`,
Nginx listens on port 80, and its own health check calls `127.0.0.1/health` from inside the
container. Those declarations should publish the port when the expected Compose project
and file are used. A refusal to EC2's own loopback address is not explained by an AWS
security-group ingress rule; that rule affects external traffic, not loopback.

**Resolution in workflow:** Make deployment deterministic by:

1. Creating a Compose command array that explicitly selects
   `"$deploy_path/docker-compose.yml"` and sets `--project-directory "$deploy_path"`.
   This avoids accidentally loading an implicit override file from a different working
   directory or using a different Compose project directory.
2. Passing `IMAGE_TAG="$deploy_sha"` to the pull and normal Compose update, preserving
   immutable commit-SHA deployment.
3. Force-recreating only Nginx with `--no-build --no-deps --force-recreate`.
4. Calling `docker compose port nginx 80` with production environment and the same SHA tag,
   failing with an explicit message unless the host port is 80.
5. Running the existing `curl` health check only after the mapping assertion passes.

**Status:** Workflow change was syntax-checked. EC2 was powered off during investigation, so
the actual container/port state has not yet been verified on the live instance. When EC2
is started, inspect `docker compose ps`, `docker compose port nginx 80`, Nginx logs, and the
health endpoint if this guard fails.

## 15. Public API needed authentication and per-user isolation

**Observed:** Gemini-backed endpoints had no authentication, and a shared Milvus collection
could otherwise return chunks across users.

**Resolution implemented:** Production mode uses Cognito JWT verification. The backend
checks the Cognito signature/JWKS, issuer, expiration and issued-at claims,
`token_use=access`, and configured app client ID, and derives identity from the verified
`sub` claim. It tags ingested chunks with that subject and applies an owner filter to vector
retrieval, lexical candidate queries, whole-document retrieval, and ingestion-job status.
A client-supplied user ID is not trusted. Cognito User Pool API access tokens are compatible
when issued by the configured pool and app client; OAuth scopes and browser redirects are not
required.

Production configuration fails closed when Cognito, Gemini, or a non-local Milvus endpoint
is missing. API docs are disabled in production. Optional Web CORS accepts explicit
origins and rejects wildcard configuration. Compose requires `APP_ENV` explicitly and the
EC2 deployment commands set it to `production`, preventing an unset environment from silently
selecting the local authentication bypass. Nginx has basic per-IP request limits for general
requests, questions, and ingestion.

**Rollout consequences and remaining work:**

- Existing chunks without an owner tag are not returned to authenticated production users;
  re-ingest documents after enabling this behavior.
- SYNQRA's frontend must migrate to Cognito and send the real access token as
  `Authorization: Bearer <token>`; an ID token or placeholder is rejected.
- Configure and verify the actual Cognito pool/region/client ID in the EC2 environment.
- Rate limiting and authentication are not a substitute for HTTPS. Terminate TLS at a
  managed load balancer/proxy and restrict EC2 ingress to the trusted proxy.
- Do not send real Cognito tokens over the current HTTP-only Nginx deployment.
- If behind an ALB, ensure rate limiting uses a trustworthy client IP and do not trust
  arbitrary forwarded headers.

The Flutter client must send `Authorization: Bearer <access_token>`; the access token must
belong to the configured Cognito user pool and app client. ID tokens are rejected.

**Verification:** Synthetic-token tests cover valid access tokens, invalid signatures,
wrong issuer, wrong client ID, ID tokens, expired tokens, missing tokens, production
configuration, cross-user job status, and owner propagation to retrieval. Milvus tests cover
owner filters for vector and whole-document queries. The full backend test suite passes.

**Status:** Backend changes are in the active deployment branch. Frontend migration, HTTPS,
production configuration, re-ingestion of pre-existing documents, and live EC2 verification
remain required.

## 16. SYNQRA endpoint did not match the deployed ports

**Observed:** SYNQRA's default endpoint was `http://localhost:8000/questions/stream`, but
the Docker Compose production stack publishes Nginx on host port 80 and does not publish
the FastAPI container's port 8000.

**Resolution/operating guidance:**

- The app's production endpoint must be the HTTPS API origin, for example
  `https://api.synqra.ai/questions/stream`; upload resolves to `/documents` on that same
  origin.
- On the Pixel, `localhost` means the phone itself. Use `adb reverse` mapped to the port
  actually published by the local service, or use the Mac's LAN IP and published port.
- iOS production traffic requires HTTPS under normal ATS policy.
- Flutter Web requires CORS for the exact deployed web origin; native Android/iOS networking
  does not use browser CORS.

The supplied EC2 address could not be contacted while the instance was turned off; that
timeout was expected, not evidence that the container was unhealthy.

**Status:** Backend/frontend port mismatch identified. Final HTTPS domain, TLS termination,
frontend production setting, and live device test remain to be completed.

## Ongoing maintenance rule

Whenever an incident is diagnosed and a fix or workaround is completed, add or update an
entry here in the same change. Record:

1. The user-visible symptom or exact error, with secrets removed.
2. The confirmed cause; label hypotheses as unconfirmed.
3. The code/configuration or operational change that resolved it.
4. The test or evidence used to verify the resolution.
5. Remaining limitations, rollout actions, or environment-only steps.

Do not include credentials, full secret values, personal documents, or sensitive provider
responses in this log. If a separate detailed guide already exists, link it and keep this
entry focused on the incident and its resolution.
