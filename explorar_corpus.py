from src.data_loading import load_corpus

df = load_corpus()

print("documentos:", len(df))

tam = df["description"].str.split().str.len()
print("tamanho medio:", round(tam.mean(), 1))

vocab = set()
df["description"].str.lower().str.split().apply(vocab.update)
print("vocabulario:", len(vocab))

print("\n--- Por gênero ---")
print(df["primary_genre"].value_counts())
