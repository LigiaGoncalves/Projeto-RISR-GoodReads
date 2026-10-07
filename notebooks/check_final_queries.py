import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import load_corpus
from src.indexing import InvertedIndex
from src.models.boolean import boolean_search

FINAL_QUERIES = [
    "second chance love story",
    "contemporary romance",
    "enemies to lovers",
    "grief and loss poems",
    "love poems about heartbreak",
    "free verse nature poetry",
    "small town romance",
    "slow burn romance",
    "sonnet collection classic",
    "friends to lovers",
]

if __name__ == "__main__":
    corpus = load_corpus()
    index = InvertedIndex()
    index.build(corpus, text_column="description")

    print("\n=== Resultado do AND booleano por consulta (checagem antes de travar) ===")
    print("(boolean_search já reconhece 'not'/'and'/'or' como operador, sem diferenciar maiúscula/minúscula)")
    for query in FINAL_QUERIES:
        n = len(boolean_search(index, query))
        print(f"  {query:45s} -> {n:,} documentos")

