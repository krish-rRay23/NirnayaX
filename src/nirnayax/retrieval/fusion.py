from typing import List, Dict, Any

def reciprocal_rank_fusion(
    vector_rankings: List[str],
    lexical_rankings: List[str],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines dense vector search ranks and sparse lexical ranks using Reciprocal Rank Fusion (RRF).
    score(d) = sum(1 / (k + rank_i(d)))
    """
    scores: Dict[str, float] = {}

    for rank, doc_id in enumerate(vector_rankings, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    for rank, doc_id in enumerate(lexical_rankings, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [{"doc_id": doc_id, "rrf_score": score} for doc_id, score in sorted_docs]
