"""
src/data_loading.py
=====================
Carregamento do corpus (Goodreads romance + poetry) via DuckDB, consultando
os .json.gz diretamente no disco sem carregar tudo na memória de uma vez.
"""

from pathlib import Path
from langdetect import detect, LangDetectException

import duckdb
import pandas as pd

DATA_DIR = Path("data/raw")  # relativo à raiz do repositório
DB_PATH = Path("data/processed/goodreads.duckdb")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

BOOKS_FILES = {
    "romance": DATA_DIR / "goodreads_books_romance.json.gz",
    "poetry": DATA_DIR / "goodreads_books_poetry.json.gz",
}


def is_english(text: str) -> bool:
    from langdetect import detect, LangDetectException
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def load_corpus(max_docs: int | None = None) -> pd.DataFrame:
    """Retorna DataFrame com book_id, title, description, primary_genre.
    Reaproveita o banco DuckDB se as tabelas já existirem (evita reprocessar
    os .gz em toda execução). A detecção de idioma também é salva no banco,
    para não rodar de novo a cada execução."""
    con = duckdb.connect(str(DB_PATH))
    tables = con.execute("SELECT table_name FROM information_schema.tables").fetchdf()["table_name"].tolist()

    if "books" in tables:
        df = con.execute("SELECT * FROM books").fetchdf()
    else:
        parts = []
        for genre, path in BOOKS_FILES.items():
            if not path.exists():
                print(f"[AVISO] {path} não encontrado, pulando {genre}.")
                continue
            posix_path = path.as_posix()
            parts.append(
                f"SELECT book_id, title, description, '{genre}' AS primary_genre "
                f"FROM read_json_auto('{posix_path}')"
            )
        if not parts:
            raise FileNotFoundError(
                "Nenhum arquivo de livros encontrado em data/raw/. "
                "Baixe goodreads_books_romance.json.gz e goodreads_books_poetry.json.gz."
            )
        query = " UNION ALL BY NAME ".join(parts)
        con.execute(f"CREATE OR REPLACE TABLE books AS {query}")
        df = con.execute("SELECT book_id, title, description, primary_genre FROM books").fetchdf()

    df = df[df["description"].fillna("").str.len() >= 20].reset_index(drop=True)
    df["book_id"] = df["book_id"].astype(int)

    # --- Detecção de idioma: só roda UMA VEZ, depois fica salva no banco ---
    if "is_english" not in df.columns:
        print("Detectando idioma (primeira vez — pode demorar alguns minutos)...")
        df["is_english"] = df["description"].apply(is_english)
        con.register("df_temp", df)
        con.execute("CREATE OR REPLACE TABLE books AS SELECT * FROM df_temp")
        con.unregister("df_temp")
        print(f"Detecção salva no banco: {df['is_english'].sum():,} documentos em inglês.")
    else:
        print(f"Idioma já detectado anteriormente ({df['is_english'].sum():,} docs em inglês) — pulando detecção.")

    df = df[df["is_english"]].reset_index(drop=True)
    con.close()

    if max_docs:
        df = df.sample(n=min(max_docs, len(df)), random_state=42).reset_index(drop=True)

    return df
