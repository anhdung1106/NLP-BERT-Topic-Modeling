"""Mô hình truyền thống: LDA (A1), LSA + K-Means (A2) (TV2)."""
from functools import lru_cache

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from src.config import (LDA_DIR, LDA_PASSES, N_TOPICS, NOISE_LABEL, SEED, TOP_N_WORDS,
                        VECTORIZER_PARAMS)
from src.preprocess import load_stopwords, make_text_tok


def top_words_by_cluster(text_tok, labels, top_n=TOP_N_WORDS):
    """Top từ của mỗi cụm theo TF-IDF trung bình (bỏ cụm nhiễu -1)."""
    vec = TfidfVectorizer(**VECTORIZER_PARAMS)
    X = vec.fit_transform(text_tok)
    vocab = vec.get_feature_names_out()
    labels = np.asarray(labels)
    result = []
    for c in sorted(set(labels) - {NOISE_LABEL}):
        mean = np.asarray(X[labels == c].mean(axis=0)).ravel()
        result.append([vocab[i] for i in mean.argsort()[::-1][:top_n]])
    return result


def run_lda(texts, k=N_TOPICS, seed=SEED):
    """LDA trên bag-of-words. Trả về (labels, topic_words, lda_model, dictionary)."""
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=3, no_above=0.5, keep_n=None)
    corpus = [dictionary.doc2bow(doc) for doc in texts]
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=LDA_PASSES,
                   random_state=seed, chunksize=2000, eval_every=None)
    labels = np.array([max(lda.get_document_topics(bow, minimum_probability=0.0),
                           key=lambda x: x[1])[0] for bow in corpus])
    topic_words = [[w for w, _ in lda.show_topic(t, topn=TOP_N_WORDS)] for t in range(k)]
    return labels, topic_words, lda, dictionary


def run_lsa_kmeans(text_tok, n_dims, k=N_TOPICS, seed=SEED):
    """TF-IDF -> TruncatedSVD -> chuẩn hóa L2 -> KMeans. Trả về (labels, topic_words, X_lsa)."""
    tfidf = TfidfVectorizer(**VECTORIZER_PARAMS).fit_transform(text_tok)
    lsa = make_pipeline(TruncatedSVD(n_components=n_dims, random_state=seed), Normalizer(copy=False))
    X_lsa = lsa.fit_transform(tfidf)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X_lsa)
    return labels, top_words_by_cluster(text_tok, labels), X_lsa


def save_lda(lda_model, dictionary, path=LDA_DIR):
    """Lưu lda.model và dictionary.dict vào models/lda/."""
    path.mkdir(parents=True, exist_ok=True)
    lda_model.save(str(path / "lda.model"))
    dictionary.save(str(path / "dictionary.dict"))


def load_lda(path=LDA_DIR):
    """Tải lại (lda_model, dictionary)."""
    return LdaModel.load(str(path / "lda.model")), Dictionary.load(str(path / "dictionary.dict"))


@lru_cache(maxsize=1)
def _stopwords():
    # đọc file stopwords 1 lần, các lần sau dùng lại
    return load_stopwords()


def predict_lda(text, lda_model, dictionary):
    """Văn bản thô -> (topic_id, xác suất). App demo dùng hàm này."""
    bow = dictionary.doc2bow(make_text_tok(text, _stopwords()).split())
    topic_id, prob = max(lda_model.get_document_topics(bow, minimum_probability=0.0), key=lambda x: x[1])
    return int(topic_id), float(prob)