"""BERTopic, 2 chế độ: HDBSCAN (tự do) và K-Means k = 6 (cố định) (TV4).

Embedding tính sẵn (embeddings/*.npy) nên BERTopic không tự nhúng văn bản (embedding_model=None).
Từ khóa c-TF-IDF lấy trên text_tok (đã tách từ, bỏ stopwords).
"""
import numpy as np
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from src.config import (BERTOPIC_DIR, BERTOPIC_VECTORIZER_PARAMS, BEST_EMBEDDING,
                        DEFAULT_MIN_CLUSTER_SIZE, N_TOPICS, NOISE_LABEL, SEED, TOP_N_WORDS,
                        UMAP_PARAMS)

MODES = ("k6", "hdbscan")


def build_bertopic(mode: str, seed: int = SEED,
                   min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE) -> BERTopic:
    """Tạo BERTopic chưa huấn luyện. mode: 'hdbscan' (tự xác định số chủ đề) hoặc 'k6' (K-Means 6 cụm)."""
    if mode not in MODES:
        raise ValueError(f"mode phải là một trong {MODES}, nhận được {mode!r}")
    umap_model = UMAP(**UMAP_PARAMS, random_state=seed)
    if mode == "hdbscan":
        cluster_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean",
                                cluster_selection_method="eom", prediction_data=True)
    else:
        cluster_model = KMeans(n_clusters=N_TOPICS, n_init=10, random_state=seed)
    # c-TF-IDF chạy trên "tài liệu gộp" của từng cụm (6 cụm ở chế độ k6), nên không lọc min_df/max_df
    # như VECTORIZER_PARAMS: min_df=3 sẽ bỏ mất từ chỉ đặc trưng cho 1-2 cụm
    vectorizer_model = CountVectorizer(**BERTOPIC_VECTORIZER_PARAMS)
    # language="multilingual": với "english" (mặc định), BERTopic xóa mọi ký tự ngoài [A-Za-z0-9 ]
    # trước c-TF-IDF, làm mất dấu tiếng Việt và dấu "_" của từ ghép. Embedding đã truyền sẵn
    # nên tham số này không làm tải thêm mô hình nào.
    return BERTopic(language="multilingual", embedding_model=None, umap_model=umap_model,
                    hdbscan_model=cluster_model, vectorizer_model=vectorizer_model,
                    top_n_words=TOP_N_WORDS, calculate_probabilities=False, verbose=False)


def fit_bertopic(docs: list[str], embeddings: np.ndarray, mode: str, seed: int = SEED,
                 min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE):
    """Huấn luyện trên docs (cột text_tok) và embedding tính sẵn. Trả về (model, labels)."""
    if len(docs) != len(embeddings):
        raise ValueError(f"Số văn bản ({len(docs)}) khác số dòng embedding ({len(embeddings)})")
    model = build_bertopic(mode, seed=seed, min_cluster_size=min_cluster_size)
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    return model, np.asarray(topics, dtype=int)


def get_topic_words(model: BERTopic, top_n: int = TOP_N_WORDS) -> list[list[str]]:
    """Top từ của mỗi chủ đề, bỏ chủ đề nhiễu -1, sắp theo số thứ tự chủ đề."""
    result = []
    for t in sorted(model.get_topics()):
        if t == NOISE_LABEL:
            continue
        words = [w for w, _ in model.get_topic(t) if w]  # BERTopic đệm "" khi chủ đề ít từ
        result.append(words[:top_n])
    return result


def reduce_noise(model: BERTopic, docs: list[str], labels, embeddings: np.ndarray) -> np.ndarray:
    """Gán văn bản nhiễu (-1) vào chủ đề gần nhất theo embedding. Không đổi model."""
    labels = np.asarray(labels, dtype=int)
    if not (labels == NOISE_LABEL).any():
        return labels.copy()
    new = model.reduce_outliers(docs, labels.tolist(), strategy="embeddings", embeddings=embeddings)
    return np.asarray(new, dtype=int)


def coords_2d(embeddings: np.ndarray, seed: int = SEED) -> np.ndarray:
    """UMAP 2 chiều (cosine) để vẽ bản đồ văn bản."""
    reducer = UMAP(n_neighbors=UMAP_PARAMS["n_neighbors"], n_components=2, min_dist=0.1,
                   metric="cosine", random_state=seed)
    return reducer.fit_transform(embeddings).astype(np.float32)


def save_bertopic(model: BERTopic, path=BERTOPIC_DIR) -> None:
    """Lưu mô hình (safetensors, kèm c-TF-IDF, không kèm mô hình embedding) cho app demo."""
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path), serialization="safetensors", save_ctfidf=True, save_embedding_model=False)


def load_bertopic(path=BERTOPIC_DIR) -> BERTopic:
    """Tải lại mô hình đã lưu bằng save_bertopic."""
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Chạy notebook 04 hoặc `git pull`.")
    return BERTopic.load(str(path))


def _topic_probs(emb: np.ndarray, model: BERTopic) -> tuple[list[int], np.ndarray]:
    """Phân bố xác suất của một văn bản trên các chủ đề (bỏ -1).

    Mô hình lưu bằng safetensors không giữ UMAP/HDBSCAN/K-Means, nên khi tải lại BERTopic chỉ dự đoán
    bằng cosine giữa embedding văn bản và vector trung bình mỗi chủ đề (topic_embeddings_). Hàm này làm
    đúng như vậy rồi lấy softmax để có xác suất, cho kết quả giống nhau trước và sau khi lưu.
    Cosine giữa các chủ đề chỉ chênh vài phần trăm và mỗi embedding có thang khác nhau (E5 ~0,8–0,9),
    nên chuẩn hóa z-score các cosine trước softmax: xác suất so sánh được giữa các embedding.
    Trả về (danh sách topic_id, xác suất tương ứng).
    """
    topic_ids = sorted(t for t in model.get_topics() if t != NOISE_LABEL)
    offset = int(model._outliers)  # topic_embeddings_ có dòng của -1 ở đầu nếu có cụm nhiễu
    centers = np.asarray(model.topic_embeddings_)[[t + offset for t in topic_ids]]
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    sims = centers @ (emb.ravel() / np.linalg.norm(emb))
    z = (sims - sims.mean()) / max(float(sims.std()), 1e-9)
    e = np.exp(z - z.max())
    return topic_ids, e / e.sum()


def predict_bertopic(text: str, model: BERTopic, key: str = BEST_EMBEDDING):
    """Văn bản thô -> (topic_id, xác suất). App demo dùng hàm này.

    Chủ đề = chủ đề có vector trung bình gần nhất (cosine), không xét cụm nhiễu; xác suất xem _topic_probs.
    """
    from src.embed import embed_query  # import muộn: chỉ tải mô hình BERT khi cần dự đoán
    topic_ids, p = _topic_probs(embed_query(text, key), model)
    i = int(p.argmax())
    return topic_ids[i], float(p[i])
