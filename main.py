from pathlib import Path

import pandas as pd
import pickle
from src.data_loading import load_corpus
from src.evaluation import build_judgment_pool, evaluate_all_models
from src.indexing import InvertedIndex
from src.models.vetorial import VectorialModel

OUTPUT_DIR = Path("ri_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TOP_K = 10


TEST_QUERIES = [
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


def main():
    corpus = load_corpus()

    INDEX_PATH = Path("data/processed/index.pkl")
    VEC_PATH = Path("data/processed/vetorial.pkl")

    if INDEX_PATH.exists() and VEC_PATH.exists():
        print("Carregando índice e modelo vetorial salvos...")
        with open(INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        with open(VEC_PATH, "rb") as f:
            vectorial_model = pickle.load(f)
    else:
        index = InvertedIndex()
        index.build(corpus, text_column="description")

        vectorial_model = VectorialModel()
        vectorial_model.fit(corpus, text_column="description")

        with open(INDEX_PATH, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(VEC_PATH, "wb") as f:
            pickle.dump(vectorial_model, f, protocol=pickle.HIGHEST_PROTOCOL)

    pool_path = OUTPUT_DIR / "judgment_pool.csv"
    if not pool_path.exists():
        print("\nGerando pool de julgamento de relevância...")
        pool_df = build_judgment_pool(index, vectorial_model, corpus, TEST_QUERIES, TOP_K)
        pool_df.to_csv(pool_path, index=False, encoding="utf-8-sig")
        print(f"Pool salvo em {pool_path}.")
        print(">>> Preencha 'relevant_0_ou_1' (1 ou 0) e rode de novo.")
        return

    pool_df = pd.read_csv(pool_path)
    if pool_df["relevant_0_ou_1"].isna().all() or (pool_df["relevant_0_ou_1"].astype(str) == "").all():
        print(f"[AVISO] {pool_path} existe mas ainda não foi julgado.")
        return

    judgments = {}
    for query, group in pool_df.groupby("query"):
        relevant_ids = set(group[group["relevant_0_ou_1"] == 1]["doc_id"])
        judgments[query] = relevant_ids

    results_df = evaluate_all_models(index, vectorial_model, TEST_QUERIES, judgments, TOP_K)
    results_df.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)

    print("\n=== Resultados da avaliação ===")
    print(results_df.to_string(index=False))
    print("\n=== Médias por modelo ===")
    print(results_df.groupby("model")[["precision", "recall", "f1"]].mean().round(3))


if __name__ == "__main__":
    main()

