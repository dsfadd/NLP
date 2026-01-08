from typing import Counter, Any
import numpy as np
from gensim.models import FastText
from sklearn.cluster import KMeans

def doc2vec_kmeans(model:FastText,sentences:list[list[str]],n_clusters=700, min_cluster_size=5):
    # Шаг 1: Кластеризация всего словаря модели
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(model.wv.vectors)

    # Метки кластеров для каждого слова (по индексам в model.wv)
    word_to_cluster = {}
    for word in model.wv.key_to_index:
        vector = model.wv[word]
        cluster_label = kmeans.predict(vector.reshape(1, -1))[0]
        word_to_cluster[word] = cluster_label

    # Шаг 2: Размеры кластеров
    cluster_sizes = Counter(word_to_cluster.values())

    # Хорошие кластеры: те, где размер >= min_cluster_size
    good_clusters = {cluster for cluster, size in cluster_sizes.items() if size >= min_cluster_size}

    # Шаг 3: Формирование doc_vectors
    doc_vectors = []
    for sentence in sentences:
        filtered_words = [word for word in sentence if
                          word in word_to_cluster and word_to_cluster[word] in good_clusters]

        if filtered_words:
            vectors = [model.wv[word] for word in filtered_words]
            doc_vector = np.mean(vectors, axis=0)
        else:
            # Если все слова отфильтрованы (редко), используйте нулевой вектор или пропустите
            doc_vector = np.zeros(model.wv.vector_size)

        doc_vectors.append(doc_vector)

    return doc_vectors
