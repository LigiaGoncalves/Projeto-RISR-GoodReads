
from src.indexing import InvertedIndex
from src.preprocessing import STOPWORDS_EN, stemmer


def boolean_search(index: InvertedIndex, query: str) -> set[int]:
    tokens = query.strip().split()
    all_docs = set(index.doc_ids)

    result: set[int] | None = None
    operator = "AND"
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.upper() in {"AND", "OR", "NOT"}:
            operator = tok.upper()
            i += 1
            continue
        if tok.lower() in STOPWORDS_EN: 
            i += 1
            continue
        term = stemmer.stem(tok.lower())
        term_docs = set(index.postings.get(term, {}).keys())

        if result is None:
            result = term_docs if operator != "NOT" else (all_docs - term_docs)
        elif operator == "AND":
            result &= term_docs
        elif operator == "OR":
            result |= term_docs
        elif operator == "NOT":
            result -= term_docs

        operator = "AND"
        i += 1

    return result or set()
