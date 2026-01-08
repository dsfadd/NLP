from gensim.models import FastText

def doc2vec_mean(model:FastText,sentences:list[list[str]]):
    vectors = []
    for sentence in sentences:
        vectors.append(model.wv.get_mean_vector(keys=sentence))
    return vectors