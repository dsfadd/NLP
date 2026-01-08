import numpy as np
from collections import Counter, defaultdict
from sklearn.cluster import MiniBatchKMeans  # лучше для больших моделей
from gensim.models import FastText

def predict_concept(model: FastText, kmeans: MiniBatchKMeans, word, word_to_concept):
    if word in word_to_concept:
        return word_to_concept[word]
    # OOV: получаем вектор и предсказываем кластер
    try:
        vector = model.wv.get_vector(word).reshape(1, -1)
        return kmeans.predict(vector)[0]
    except KeyError:
        return -1  # если совсем нет вектора

def get_single_embedding(model,
                         kmeans,
                         num_concepts,
                         doc_words,
                         word_to_concept,
                         concept_idf,
                         normalize,
                         default_idf=None):

        concept_counter = Counter()

        for word in doc_words:
            concept = predict_concept(model=model, kmeans=kmeans, word=word, word_to_concept=word_to_concept)
            if concept != -1:
                concept_counter[concept] += 1

        if not concept_counter:
            return np.zeros(num_concepts)

        embedding = np.zeros(num_concepts)
        for concept, cf in concept_counter.items():  # cf = concept frequency
            idf = concept_idf.get(concept, default_idf if default_idf is not None else np.log(1))  # безопасно
            embedding[concept] = cf * idf  # CF-IDF вес

        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

        return embedding

def get_boc_embeddings(sentences: list[list[str]], model: FastText, num_concepts: int = 800, normalize: bool = True):
    """
    Возвращает список BOC-эмбеддингов для каждого предложения.
    """
    # 1. Кластеризация ВСЕХ слов из vocab модели
    if len(model.wv.vectors) < num_concepts:
        print(
            f"Внимание: слов в модели ({len(model.wv.vectors)}) меньше, чем кластеров ({num_concepts}). Уменьшаю num_concepts.")
        num_concepts = max(1, len(model.wv.vectors))

    kmeans = MiniBatchKMeans(n_clusters=num_concepts, random_state=42, batch_size=1024)
    kmeans.fit(model.wv.vectors)

    # 2. Словарь слово → концепт для всех слов в vocab
    word_to_concept = {}
    for word in model.wv.key_to_index:
        idx = model.wv.key_to_index[word]
        vector = model.wv.vectors[idx].reshape(1, -1)
        concept_id = kmeans.predict(vector)[0]
        word_to_concept[word] = concept_id

    total_documents = len(sentences)
    concept_doc_frequency = defaultdict(int)
    for doc_words in sentences:
        doc_concepts = set()
        for word in doc_words:
            concept = predict_concept(model, kmeans, word, word_to_concept)
            if concept != -1:
                doc_concepts.add(concept)
        for concept in doc_concepts:
            concept_doc_frequency[concept] += 1

    concept_idf = {}
    for concept, df in concept_doc_frequency.items():
        concept_idf[concept] = np.log(total_documents / (df + 1))

    default_idf = np.log(total_documents)

    embeddings = [get_single_embedding(model=model, kmeans=kmeans, num_concepts=num_concepts, doc_words=sentence,
                                       word_to_concept=word_to_concept, normalize=normalize,default_idf=default_idf,concept_idf=concept_idf) for sentence in sentences]

    return embeddings
