"""
=================================
Exploração de vocabulário real do corpus — usei antes de fixar as consultas
de teste em main.py, para garantir que existiam documentos associados.
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # raiz do repo

from src.data_loading import load_corpus
from src.indexing import InvertedIndex
from src.preprocessing import preprocess

TOP_N = 40


def top_terms_by_genre(corpus: pd.DataFrame, genre: str, n: int = TOP_N) -> list[tuple[str, int]]:
    subset = corpus[corpus["primary_genre"] == genre]
    counter = Counter()
    for desc in subset["description"].dropna():
        counter.update(preprocess(desc))
    return counter.most_common(n)


def bigrams_by_genre(corpus: pd.DataFrame, genre: str, n: int = TOP_N) -> list[tuple[str, int]]:
    subset = corpus[corpus["primary_genre"] == genre]
    counter = Counter()
    for desc in subset["description"].dropna():
        tokens = preprocess(desc)
        counter.update([" ".join(b) for b in zip(tokens, tokens[1:])])
    return counter.most_common(n)


def document_frequency_check(index: InvertedIndex, candidate_terms: list[str]) -> pd.DataFrame:
    rows = []
    for term in candidate_terms:
        stemmed = preprocess(term)
        dfs = [index.document_frequency(t) for t in stemmed]
        rows.append({
            "termo_original": term,
            "termo_processado": " ".join(stemmed),
            "df_por_token": dfs,
            "pct_do_corpus": [round(d / index.N * 100, 2) for d in dfs],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    corpus = load_corpus()

    print("\n=== TOP TERMOS — ROMANCE ===")
    for term, count in top_terms_by_genre(corpus, "romance"):
        print(f"  {term:20s} {count:,}")

    print("\n=== TOP TERMOS — POETRY ===")
    for term, count in top_terms_by_genre(corpus, "poetry"):
        print(f"  {term:20s} {count:,}")

    print("\n=== TOP BIGRAMAS — ROMANCE ===")
    for bigram, count in bigrams_by_genre(corpus, "romance"):
        print(f"  {bigram:30s} {count:,}")

    print("\n=== TOP BIGRAMAS — POETRY ===")
    for bigram, count in bigrams_by_genre(corpus, "poetry"):
        print(f"  {bigram:30s} {count:,}")

    CANDIDATE_QUERIES = [
        "marriage", "second chance", "enemies to lovers", "small town",
        "free verse", "grief", "sonnet", "heartbreak", "contemporary",
    ]

    index = InvertedIndex()
    index.build(corpus, text_column="description")

    print("\n=== CHECAGEM DE DOCUMENT FREQUENCY ===")
    print(document_frequency_check(index, CANDIDATE_QUERIES).to_string(index=False))
