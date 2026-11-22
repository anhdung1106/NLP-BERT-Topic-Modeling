"""Trích xuất embedding B0–B5 và chạy A3 (BERT + K-Means) (TV3).

Chạy từ thư mục gốc repo:
    python -m src.embed                       # tính các embedding chưa có file
    python -m src.embed --keys b3_minilm      # chỉ tính một vài khóa
    python -m src.embed --overwrite           # tính lại, ghi đè file cũ
"""
import argparse
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import (ARTICLES_CSV, EMB_DIR, EMB_TIME_CSV, EMBEDDINGS, MAX_LENGTH,
                        N_TOPICS, SEED, VECTORIZER_PARAMS)
from src.preprocess import load_articles, make_text_bert, segment

TIME_COLUMNS = ["embedding", "n_docs", "dim", "time_s"]


def _l2(X: np.ndarray) -> np.ndarray:
    """Chuẩn hóa L2 từng dòng, ép về float32."""
    return normalize(np.asarray(X, dtype=np.float32), norm="l2").astype(np.float32)


# ---------------------------------------------------------------- B0
def embed_tfidf_svd(text_tok: list[str], dim: int = 100, seed: int = SEED) -> np.ndarray:
    """B0: TF-IDF (text_tok) -> TruncatedSVD `dim` chiều -> chuẩn hóa L2."""
    tfidf = TfidfVectorizer(**VECTORIZER_PARAMS).fit_transform(text_tok)
    dim = min(dim, tfidf.shape[1] - 1)  # tránh lỗi khi chạy trên dữ liệu mẫu nhỏ
    X = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(tfidf)
    return _l2(X)


# ---------------------------------------------------------------- B1–B5
@lru_cache(maxsize=None)
def _load_hf(model_name: str):
    """Tải tokenizer + mô hình transformers (B1, B2). Chỉ tải 1 lần mỗi mô hình."""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    return tokenizer, model


@lru_cache(maxsize=None)
def _load_sbert(model_name: str):
    """Tải SentenceTransformer (B3, B4, B5). Chỉ tải 1 lần mỗi mô hình."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    model.max_seq_length = MAX_LENGTH
    return model


def _embed_transformers(texts: list[str], model_name: str, pooling: str, batch_size: int) -> np.ndarray:
    """BERT nguyên bản: pooling 'cls' (vector token đầu) hoặc 'mean' (trung bình token thật)."""
    import torch
    tokenizer, model = _load_hf(model_name)
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
            hidden = model(**enc).last_hidden_state              # (batch, token, dim)
            if pooling == "cls":
                vec = hidden[:, 0]
            else:                                                # mean: bỏ token đệm bằng attention mask
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            out.append(vec.cpu().numpy())
            if (i // batch_size) % 10 == 0:
                print(f"    {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.vstack(out)


def embed_texts(texts: list[str], key: str, batch_size: int = 32) -> np.ndarray:
    """B1–B5 theo EMBEDDINGS[key]. Trả về (n, d) float32, đã chuẩn hóa L2."""
    cfg = EMBEDDINGS[key]
    if cfg["model"] is None:
        raise ValueError(f"{key} không dùng BERT: gọi embed_tfidf_svd thay cho embed_texts")
    prefix = cfg.get("prefix", "")
    texts = [prefix + t for t in texts]
    if cfg["pooling"] == "sbert":
        model = _load_sbert(cfg["model"])
        X = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=False)
    else:
        X = _embed_transformers(texts, cfg["model"], cfg["pooling"], batch_size)
    return _l2(X)


# ---------------------------------------------------------------- tính, lưu, đọc
def _warmup(key: str) -> None:
    """Tải mô hình (nếu chưa có) và chạy thử 1 câu, để thời gian đo không lẫn thời gian tải mạng."""
    if EMBEDDINGS[key]["model"] is not None:
        embed_texts(["khởi động mô hình"], key)


def _save_times(new_rows: list[dict], time_csv=EMB_TIME_CSV) -> pd.DataFrame:
    """Ghi thêm/ghi đè dòng thời gian theo khóa `embedding`, trả về bảng đầy đủ."""
    table = pd.DataFrame(new_rows, columns=TIME_COLUMNS)
    if time_csv.exists():
        old = pd.read_csv(time_csv)
        table = pd.concat([old, table], ignore_index=True)
    table = table.drop_duplicates("embedding", keep="last")
    time_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(time_csv, index=False, encoding="utf-8")
    return table


def compute_all(df: pd.DataFrame, keys=None, overwrite: bool = False,
                out_dir=EMB_DIR, time_csv=EMB_TIME_CSV) -> pd.DataFrame:
    """Tính embedding cho các khóa trong `keys` (mặc định tất cả), lưu <out_dir>/<key>.npy
    và ghi thời gian vào EMB_TIME_CSV (cột embedding, n_docs, dim, time_s).

    File đã có thì bỏ qua (chạy tiếp được sau khi bị ngắt); overwrite=True để tính lại.
    out_dir, time_csv: chỉ đổi khi chạy thử trên dữ liệu mẫu, để không làm bẩn thư mục chính thức.
    """
    keys = list(keys) if keys else list(EMBEDDINGS)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(time_csv) if time_csv.exists() else pd.DataFrame(columns=TIME_COLUMNS)
    for key in keys:
        path = out_dir / f"{key}.npy"
        if path.exists() and not overwrite:
            print(f"[{key}] đã có {path.name}, bỏ qua (overwrite=True để tính lại)")
            continue
        cfg = EMBEDDINGS[key]
        texts = df[cfg["text_col"]].tolist()
        print(f"[{key}] {len(texts)} văn bản, cột {cfg['text_col']}")
        _warmup(key)
        t0 = time.perf_counter()
        if cfg["model"] is None:
            X = embed_tfidf_svd(texts, dim=cfg.get("dim", 100))
        else:
            X = embed_texts(texts, key)
        time_s = time.perf_counter() - t0
        np.save(path, X)
        print(f"[{key}] xong: {X.shape}, {time_s:.1f}s")
        # ghi sau mỗi khóa để bị ngắt giữa chừng không mất kết quả
        table = _save_times([{"embedding": key, "n_docs": int(X.shape[0]), "dim": int(X.shape[1]),
                              "time_s": round(time_s, 4)}], time_csv)
    return table


def load_embedding(key: str, n_expected: int | None = None) -> np.ndarray:
    """Đọc <EMB_DIR>/<key>.npy và kiểm tra số dòng bằng số bài trong articles.csv."""
    if key not in EMBEDDINGS:
        raise KeyError(f"Khóa không hợp lệ: {key}. Hợp lệ: {list(EMBEDDINGS)}")
    path = EMB_DIR / f"{key}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Chạy `python -m src.embed` hoặc `git pull`.")
    X = np.load(path)
    if n_expected is None:
        n_expected = len(pd.read_csv(ARTICLES_CSV, usecols=["id"], encoding="utf-8"))
    if X.shape[0] != n_expected:
        raise ValueError(f"{path.name} có {X.shape[0]} dòng nhưng articles.csv có {n_expected} bài. "
                         "Embedding được tính trên bản dữ liệu khác: tính lại hoặc pull bản mới.")
    return X


def embed_query(text: str, key: str) -> np.ndarray:
    """Văn bản thô -> embedding (1, d), xử lý giống lúc tính embedding. App demo dùng hàm này."""
    cfg = EMBEDDINGS[key]
    if cfg["model"] is None:
        raise ValueError("B0 (TF-IDF + SVD) không nhúng được văn bản mới vì không lưu mô hình; dùng b1–b5")
    t = make_text_bert(text, "")                 # đoạn nhập vào coi như tiêu đề, không có sapo
    if cfg["text_col"] == "text_bert_seg":       # B4 cần tách từ
        t = segment(t)
    return embed_texts([t], key)


# ---------------------------------------------------------------- A3
def run_bert_kmeans(X: np.ndarray, k: int = N_TOPICS, seed: int = SEED) -> np.ndarray:
    """A3: K-Means (n_init=10) trên embedding đã chuẩn hóa. Trả về mảng nhãn."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tính embedding B0–B5")
    parser.add_argument("--keys", nargs="+", choices=list(EMBEDDINGS), help="chỉ tính các khóa này")
    parser.add_argument("--overwrite", action="store_true", help="tính lại, ghi đè file cũ")
    args = parser.parse_args()
    df = load_articles()
    print(f"Đọc {len(df)} bài từ articles.csv")
    print(compute_all(df, keys=args.keys, overwrite=args.overwrite).to_string(index=False))


if __name__ == "__main__":
    main()
