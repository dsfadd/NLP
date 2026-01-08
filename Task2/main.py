import gzip
import re
from Objects.text import Text
from gensim.models import FastText
from typing import Iterator, List
from cluster import doc2vec_kmeans
from boc import get_boc_embeddings
from mean import doc2vec_mean
from test import test_svm


# Чтение файла данных
def read_texts(fn: str="Objects/news.txt.gz") -> Iterator[Text]:
    with gzip.open(fn, "rt", encoding="utf-8") as f:
        for line in f:
            yield Text(*line.strip().split("\t"))

from nltk.corpus import stopwords
stop_words = set(stopwords.words('russian'))
from pymorphy3 import MorphAnalyzer
morph = MorphAnalyzer()

# Разбиение текста на слова
def tokenize_text(text: str) -> List[str]:
    words = re.findall(r'\b\w+\b', text.lower())
    lemmas = [morph.parse(word)[0].normal_form for word in words if word not in stop_words]

    return lemmas

# каждый текст - набор слов через пробел
texts = list(read_texts())

sentences = [tokenize_text(text.text) for text in texts]
labels=[t.label for t in texts]

# Инициализация
try:
    model = FastText.load('w2v_vectors.model')
except FileNotFoundError:
    model = FastText(sentences=sentences)
    model.save('w2v_vectors.model')

# vecs=doc2vec_kmeans(model,sentences)
# vecs=doc2vec_mean(model,sentences)
vecs=get_boc_embeddings(sentences,model)
test_svm(vecs, labels)
