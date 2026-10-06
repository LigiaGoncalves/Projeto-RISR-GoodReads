import math
from collections import defaultdict

from src.indexing import InvertedIndex
from src.preprocessing import preprocess


def bm25_search(
    index: InvertedIndex, query: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75
) -> list[tuple[int, float]]:
    query_terms = preprocess(query)
    if not query_terms:
        return []

    scores: dict[int, float] = defaultdict(float)

    for term in set(query_terms):
        df_term = index.document_frequency(term)
        if df_term == 0:
            continue
        idf = math.log(1 + (index.N - df_term + 0.5) / (df_term + 0.5))

        postings = index.postings.get(term, {})
        for doc_id, tf in postings.items():
            doc_len = index.doc_lengths.get(doc_id, 1)
            norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / index.avg_doc_length))
            scores[doc_id] += idf * norm

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
