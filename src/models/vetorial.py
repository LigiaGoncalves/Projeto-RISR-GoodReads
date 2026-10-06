import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import preprocess


class VectorialModel:
    def __init__(self):
        # tokenizer=preprocess reaproveita o pipeline principal;
        # lowercase=False e token_pattern=None porque o preprocess já cuida disso.
        self.vectorizer = TfidfVectorizer(tokenizer=preprocess, lowercase=False, token_pattern=None)
        self.doc_ids: list[int] = []
        self.matrix = None

    def fit(self, corpus_df: pd.DataFrame, text_column: str = "description") -> None:
        print("Ajustando o modelo vetorial (TF-IDF, sklearn)...")
        self.doc_ids = corpus_df["book_id"].tolist()
        texts = corpus_df[text_column].fillna("").tolist()
        self.matrix = self.vectorizer.fit_transform(texts)
        print(f"Modelo vetorial ajustado: matriz {self.matrix.shape[0]:,} docs x "
              f"{self.matrix.shape[1]:,} termos.")

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        if self.matrix is None:
            raise RuntimeError("Chame fit() antes de search().")
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        return [(self.doc_ids[i], float(sims[i])) for i in top_idx if sims[i] > 0]
