# RAG Evaluation Guide

The evaluation harness checks whether retrieval and answer-generation changes improve the
project's results. It has two modes:

- **Offline** is deterministic and is the default. It runs with the fictional fixture vectors
  and existing hybrid retrieval code; it does not call Gemini, Milvus, or the network.
- **Live** explicitly calls Gemini and Milvus using the local environment configuration. It
  indexes only the fictional fixture corpus into a temporary collection, evaluates answers,
  and removes the collection afterward.

## Run offline

Install the project's normal development dependencies, then run:

```sh
python -m app.evaluation --mode offline
```

The default report is `eval-results/latest.json`; this directory is ignored by Git. CI runs
this offline evaluator for each validated branch and does not need API credentials.

Change retrieval parameters when comparing algorithm settings:

```sh
python -m app.evaluation --mode offline \
  --candidates 20 \
  --top-k 5 \
  --rrf-k 60 \
  --output eval-results/current.json
```

The checked-in `evaluation/baseline.json` is the initial offline baseline. Compare a later
run against it:

```sh
python -m app.evaluation --mode offline \
  --output eval-results/current.json \
  --compare evaluation/baseline.json
```

Positive deltas mean the metric increased; whether that is better depends on the metric.
For example, higher retrieval recall and citation recall are generally desirable. A strict
offline report is repeatable except for timestamp and Git commit metadata. Comparisons require
matching evaluator mode, dataset, and `--top-k` cutoff; retrieval settings such as candidate
count and RRF constant may differ intentionally.

## Run live

Live evaluation requires the same Gemini and Milvus configuration as the application:

```sh
python -m app.evaluation --mode live --output eval-results/live.json
```

Each run generates an unpredictable collection name with the `rag_eval_` prefix. The
evaluator refuses names that do not match that format, refuses to use the configured
application collection, and drops the temporary collection after the run. Only the invented
fixture corpus is indexed. The live run can consume embedding and generation quota.

Add model-based scoring only when desired:

```sh
python -m app.evaluation --mode live --judge \
  --output eval-results/live-judged.json
```

The judge makes an additional Gemini request per case and returns scores from 1 to 5 for
faithfulness and relevance. Treat these as model estimates, not deterministic ground truth.
The live report includes timing and model configuration, but excludes credentials and the
Milvus URI.

## Metrics

### Retrieval

- **Hit Rate@K**: fraction of questions where at least one relevant page appears in the top K.
- **Recall@K**: relevant pages retrieved divided by all relevant pages annotated for the case.
- **MRR@K**: reciprocal rank of the first relevant page, averaged across questions.
- **nDCG@K**: ranking quality using graded relevance labels; highly relevant pages contribute
  more gain than partially relevant pages.

The golden set labels relevance from 0 to 3, with 0 implicit for unlisted pages. Ground truth
is page-level so the same cases can evaluate different chunking strategies even when chunk
IDs change.

### Answers and citations

- **Answer token precision/recall/F1**: normalized token overlap between a live answer and its
  curated reference answer. This is a simple, explainable signal, not a complete semantic
  correctness measure.
- **Citation precision/recall/F1**: compares the response's cited document/page pairs with the
  expected supporting pages in the golden case.
- **LLM judge scores**: optional 1–5 faithfulness and relevance estimates from Gemini.

### Timing

Live case reports include retrieval and generation seconds. The aggregate run timings include
fixture ingestion and the overall live run. Timing depends on provider/network load and should
not be treated as deterministic.

## Golden dataset

Edit [`evaluation/golden_dataset.json`](https://github.com/nithishkumar2022020/RAG_BACKEND/blob/main/evaluation/golden_dataset.json) to add cases.
Use invented example content only. Each document has a stable fixture ID, filename, and
page-level text. Offline mode additionally uses fixed vectors for predictable dense ranking;
live mode ignores those fixture vectors and generates embeddings.

Each case should include:

- a unique `id` and representative question
- allowed `document_ids`
- offline `query_vector` with the same dimension as page vectors
- one or more `relevant_pages`, each with a graded relevance from 1 to 3
- a concise reference answer
- expected citation document/page pairs

Add cases for direct factual questions, paraphrases, multiple documents, missing facts, and
the behavior being changed. A metric only reflects the quality and coverage of its human-made
gold labels.

Do not add real uploaded PDFs, secrets, personal information, or provider responses containing
sensitive data. Generated reports are ignored by Git by default; review them before explicitly
saving a baseline.
