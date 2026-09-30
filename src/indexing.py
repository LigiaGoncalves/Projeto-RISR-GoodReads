from collections import Counter, defaultdict

import pandas as pd

from src.preprocessing import preprocess


class InvertedIndex:
    def __init__(self):
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_lengths: dict[int, int] = {}
        self.doc_ids: list[int] = []
        self.N: int = 0
        self.avg_doc_length: float = 0.0

    def build(self, df: pd.DataFrame, text_column: str = "description") -> None:
        print("Construindo índice invertido...")
        total_len = 0
        for i, row in enumerate(df.itertuples(index=False)):
            doc_id = getattr(row, "book_id")
            text = getattr(row, text_column)
            tokens = preprocess(text)
            term_freqs = Counter(tokens)

            for term, tf in term_freqs.items():
                self.postings[term][doc_id] = tf

            self.doc_lengths[doc_id] = len(tokens)
            total_len += len(tokens)
            self.doc_ids.append(doc_id)

            if (i + 1) % 20000 == 0:
                print(f"  ...{i+1:,} documentos indexados")

        self.N = len(self.doc_ids)
        self.avg_doc_length = total_len / max(self.N, 1)
        print(f"Índice construído: {self.N:,} documentos, {len(self.postings):,} termos únicos "
              f"(tamanho médio de documento: {self.avg_doc_length:.1f} tokens)")

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, {}))
