

import math

import pandas as pd

from src.indexing import InvertedIndex
from src.models.boolean import boolean_search
from src.models.probabilistico import bm25_search
from src.models.vetorial import VectorialModel


def build_judgment_pool(
    index: InvertedIndex,
    vectorial_model: VectorialModel,
    corpus: pd.DataFrame,
    queries: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    corpus_lookup = corpus.set_index("book_id")[["title", "description", "primary_genre"]]
    rows = []

    for query in queries:
        tfidf_results = {doc_id for doc_id, _ in vectorial_model.search(query, top_k)}
        bm25_results = {doc_id for doc_id, _ in bm25_search(index, query, top_k)}
        boolean_results = set(sorted(boolean_search(index, query))[:top_k])

        pooled_ids = tfidf_results | bm25_results | boolean_results

        for doc_id in pooled_ids:
            if doc_id not in corpus_lookup.index:
                continue
            info = corpus_lookup.loc[doc_id]
            rows.append({
                "query": query,
                "doc_id": doc_id,
                "title": info["title"],
                "description_preview": str(info["description"])[:2000],
                "genre": info["primary_genre"],
                "relevant_0_ou_1": "",
            })

    pool_df = pd.DataFrame(rows)
    pool_df = pool_df.sample(frac=1, random_state=123).reset_index(drop=True)
    return pool_df


def precision_recall_f1(retrieved: list[int], relevant: set[int]) -> tuple[float, float, float]:
    if not retrieved:
        return 0.0, 0.0, 0.0
    retrieved_set = set(retrieved)
    tp = len(retrieved_set & relevant)
    precision = tp / len(retrieved_set)
    recall = tp / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def ndcg_at_k(ranked_doc_ids: list[int], relevant: set[int], k: int = 10) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        rel = 1 if doc_id in relevant else 0
        dcg += rel / math.log2(i + 2)

    ideal_rels = sorted([1] * min(len(relevant), k) + [0] * max(0, k - len(relevant)), reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_all_models(
    index: InvertedIndex,
    vectorial_model: VectorialModel,
    queries: list[str],
    judgments: dict[str, set[int]],
    top_k: int = 10,
) -> pd.DataFrame:
    rows = []
    for query in queries:
        relevant = judgments.get(query, set())
        if not relevant:
            print(f"[AVISO] Sem julgamentos para a query '{query}', pulando.")
            continue

        bool_results = sorted(boolean_search(index, query))[:top_k]
        tfidf_results = [doc_id for doc_id, _ in vectorial_model.search(query, top_k)]
        bm25_results = [doc_id for doc_id, _ in bm25_search(index, query, top_k)]

        for model_name, results in [
            ("Booleano", bool_results),
            ("Vetorial (TF-IDF)", tfidf_results),
            ("Probabilístico (BM25)", bm25_results),
        ]:
            p, r, f1 = precision_recall_f1(results, relevant)
            ndcg = ndcg_at_k(results, relevant, top_k) if model_name != "Booleano" else None
            rows.append({
                "query": query, "model": model_name,
                "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            })

    return pd.DataFrame(rows)
