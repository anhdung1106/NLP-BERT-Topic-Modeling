"""Bộ chỉ số đánh giá dùng chung (TV2)."""
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from src.config import NOISE_LABEL, RESULTS_CSV, TABLES_DIR, TOP_N_WORDS

RESULT_COLUMNS = ["method", "embedding", "mode", "param", "seed", "n_topics", "pct_noise",
                  "ari", "nmi", "purity", "cv", "diversity", "time_s"]
KEY_COLUMNS = ["method", "embedding", "mode", "param", "seed"]


def purity(labels_true, labels_pred) -> float:
    m = contingency_matrix(labels_true, labels_pred)  # hàng: nhãn thật, cột: cụm
    return float(m.max(axis=0).sum() / m.sum())


def topic_diversity(topic_words, top_n=TOP_N_WORDS) -> float:
    if not topic_words:
        return 0.0
    words = [w for topic in topic_words for w in topic[:top_n]]
    return len(set(words)) / (len(topic_words) * top_n)


def coherence_cv(topic_words, texts) -> float:
    dictionary = Dictionary(texts)
    topics = [[w for w in t if w in dictionary.token2id] for t in topic_words]
    topics = [t for t in topics if len(t) >= 2]   # chủ đề còn < 2 từ thì bỏ
    if not topics:
        return float("nan")
    cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary,
                        coherence="c_v", processes=1)
    return float(cm.get_coherence())


def evaluate(labels_pred, labels_true, topic_words, texts, ignore_noise=True) -> dict:
    labels_pred, labels_true = np.asarray(labels_pred), np.asarray(labels_true)
    is_noise = labels_pred == NOISE_LABEL
    mask = ~is_noise if ignore_noise else np.ones(len(labels_pred), dtype=bool)
    yt, yp = labels_true[mask], labels_pred[mask]
    return {
        "ari": adjusted_rand_score(yt, yp),
        "nmi": normalized_mutual_info_score(yt, yp),
        "purity": purity(yt, yp),
        "cv": coherence_cv(topic_words, texts),
        "diversity": topic_diversity(topic_words),
        "n_topics": int(len(np.unique(labels_pred[~is_noise]))),
        "pct_noise": float(is_noise.mean()),
    }


def make_row(method, embedding, mode, param, seed, scores: dict, time_s: float) -> dict:
    row = {"method": method, "embedding": embedding, "mode": mode,
           "param": param or "", "seed": int(seed), "time_s": time_s}
    row.update({k: scores[k] for k in ["n_topics", "pct_noise", "ari", "nmi",
                                       "purity", "cv", "diversity"]})
    row = {c: row[c] for c in RESULT_COLUMNS}
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}


def save_results(rows: list[dict]) -> None:
    new = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if RESULTS_CSV.exists():
        old = pd.read_csv(RESULTS_CSV, keep_default_na=False)
        new = pd.concat([old, new], ignore_index=True)
    new["seed"] = new["seed"].astype(int)
    new["param"] = new["param"].astype(str)
    new = new.drop_duplicates(KEY_COLUMNS, keep="last")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    new.to_csv(RESULTS_CSV, index=False, encoding="utf-8")