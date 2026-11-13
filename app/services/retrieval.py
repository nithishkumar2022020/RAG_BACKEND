import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def reciprocal_rank_fusion(rankings: list[list[dict]], constant: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            chunk_id = str(chunk["id"])
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (constant + rank)
            chunks[chunk_id] = chunk
    return sorted(chunks.values(), key=lambda chunk: scores[str(chunk["id"])], reverse=True)


def hybrid_search(
    query: str,
    dense_results: list[dict],
    chunks: list[dict],
    candidates: int = 20,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    if not chunks:
        return []
    corpus = [tokenize(chunk["text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    lexical_results = [chunks[index] for index in scores.argsort()[::-1][:candidates]]
    fused = reciprocal_rank_fusion([dense_results[:candidates], lexical_results], rrf_k)
    score_by_id = {str(chunk["id"]): float(scores[index]) for index, chunk in enumerate(chunks)}
    rrf_order = {str(chunk["id"]): rank for rank, chunk in enumerate(fused)}
    return sorted(
        fused,
        key=lambda chunk: (score_by_id.get(str(chunk["id"]), 0), -rrf_order[str(chunk["id"])]),
        reverse=True,
    )[:top_k]
