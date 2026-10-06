"""Testes dos três modelos de ranking sobre um corpus sintético pequeno."""

import pandas as pd
import pytest

from src.indexing import InvertedIndex
from src.models.boolean import boolean_search
from src.models.probabilistico import bm25_search
from src.models.vetorial import VectorialModel


@pytest.fixture
def small_corpus():
    return pd.DataFrame([
        {"book_id": 1, "title": "A", "description": "A billionaire falls in love with a small town baker.", "primary_genre": "romance"},
        {"book_id": 2, "title": "B", "description": "Enemies to lovers slow burn romance with banter.", "primary_genre": "romance"},
        {"book_id": 3, "title": "C", "description": "A collection of free verse poems about grief and loss.", "primary_genre": "poetry"},
        {"book_id": 4, "title": "D", "description": "Sonnets about love and heartbreak, a classic collection.", "primary_genre": "poetry"},
    ])


@pytest.fixture
def index(small_corpus):
    idx = InvertedIndex()
    idx.build(small_corpus, text_column="description")
    return idx


def test_boolean_and(index):
    result = boolean_search(index, "love AND billionaire")
    assert result == {1}


def test_boolean_not(index):
    result = boolean_search(index, "love NOT poem")
    assert 3 not in result  # doc 3 fala de "poems", não deve ser afetado por NOT poem incorretamente
    assert 1 in result


def test_bm25_ranks_relevant_first(index):
    results = bm25_search(index, "small town romance", top_k=4)
    result_ids = [doc_id for doc_id, _ in results]
    assert result_ids[0] == 1  # doc 1 é o mais relevante para essa query


def test_vectorial_search(small_corpus):
    model = VectorialModel()
    model.fit(small_corpus, text_column="description")
    results = model.search("grief and loss", top_k=4)
    result_ids = [doc_id for doc_id, _ in results]
    assert result_ids[0] == 3
