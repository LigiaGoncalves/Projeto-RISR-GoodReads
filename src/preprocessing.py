import re
import nltk

for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

NEGATION_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "nowhere", "neither",
    "without", "cannot", "can't", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "isn't", "aren't", "wasn't", "weren't",
}

STOPWORDS_EN = set(stopwords.words("english")) - NEGATION_WORDS

stemmer = SnowballStemmer("english")


def preprocess(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.lower()
    text = text.replace("’", "'")                 # apóstrofo tipográfico -> ASCII
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"n't\b", " not", text)         # don't, isn't, didn't...
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    tokens = word_tokenize(text)

    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in STOPWORDS_EN or t in NEGATION_WORDS]
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def preprocess_to_string(text: str) -> str:
    """Retorna string única (tokens separados por espaço)"""
    return " ".join(preprocess(text))
