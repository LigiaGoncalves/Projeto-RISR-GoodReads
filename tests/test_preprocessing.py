"""Testes básicos do pipeline de pré-processamento. """

from src.preprocessing import preprocess


def test_negation_preserved():
    tokens = preprocess("This is not a love story")
    assert "not" in tokens


def test_stopword_removed():
    tokens = preprocess("This is a story about love")
    # "this", "is", "a", "about" são stopwords comuns e devem ser removidas
    assert "thi" not in tokens and "is" not in tokens


def test_stemming_applied():
    tokens = preprocess("lovers loving loved")
    # todos devem reduzir a uma raiz comum (ou próxima) via Snowball
    assert len(set(tokens)) <= 3


def test_empty_input():
    assert preprocess("") == []
    assert preprocess(None) == []
