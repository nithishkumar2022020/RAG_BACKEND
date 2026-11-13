from app.services.retrieval import hybrid_search, reciprocal_rank_fusion


def chunk(chunk_id: str, text: str) -> dict:
    return {"id": chunk_id, "text": text}


def test_rrf_promotes_items_ranked_by_both_retrievers():
    dense = [chunk("a", "alpha"), chunk("b", "beta")]
    lexical = [chunk("b", "beta"), chunk("c", "gamma")]

    assert [item["id"] for item in reciprocal_rank_fusion([dense, lexical])] == ["b", "a", "c"]


def test_bm25_reranks_fused_candidates_and_limits_top_k():
    chunks = [
        chunk("a", "apple fruit"),
        chunk("b", "apple apple pie"),
        chunk("c", "banana bread"),
    ]

    result = hybrid_search("apple", [chunks[2], chunks[0]], chunks, candidates=3, top_k=1)

    assert [item["id"] for item in result] == ["b"]


def test_hybrid_search_returns_at_most_five_results():
    chunks = [chunk(str(index), f"topic {index}") for index in range(8)]

    assert len(hybrid_search("topic", chunks, chunks, candidates=8)) == 5
