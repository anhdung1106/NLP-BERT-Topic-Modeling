# Quy ước code

Tài liệu **bắt buộc tuân theo** cho 5 thành viên lập trình. Mục đích: 5 người code song song mà khi ghép lại không bị lệch tên hàm, tên biến, tên cột, tên file.

- Tên hàm, tham số, giá trị trả về trong tài liệu này là **cố định**. Muốn đổi: báo cả nhóm trước, sửa tài liệu này trước rồi mới sửa code.
- Được thêm hàm phụ (bắt đầu bằng `_`, ví dụ `_fetch_page`) trong file của mình, miễn không đổi các hàm đã quy định.
- Phân công công việc: [PHAN_CONG.md](PHAN_CONG.md) · Cách dùng Git: [HUONG_DAN_GITHUB.md](HUONG_DAN_GITHUB.md)

---

## 1. Quy tắc chung

| Hạng mục | Quy tắc | Ví dụ |
|---|---|---|
| Tên hàm, biến | Tiếng Anh, `snake_case` | `load_articles`, `labels_pred` |
| Hằng số | Chữ in hoa, chỉ khai báo trong `src/config.py` | `SEED`, `ARTICLES_CSV` |
| Tên file, thư mục | Không dấu, chữ thường, `_` | `01_du_lieu.ipynb`, `b3_minilm.npy` |
| Chú thích trong code | Tiếng Việt, ngắn | `# bỏ bài dưới 150 từ` |
| Đường dẫn | **Luôn lấy từ `src/config.py`**, không gõ cứng đường dẫn trong code hay notebook | `from src.config import ARTICLES_CSV` |
| Seed | Mọi hàm có yếu tố ngẫu nhiên phải nhận tham số `seed` | `run_lda(..., seed=SEED)` |
| Import | Notebook và app import từ `src` | `from src.metrics import evaluate` |
| Chạy script | Chạy từ thư mục gốc repo bằng `python -m` | `python -m src.crawl` |

**Notebook:** ô đầu tiên của mọi notebook là:

```python
import sys; sys.path.append("..")
from src.config import *
```

---

## 2. `src/config.py` 

Mọi đường dẫn, seed, danh sách chủ đề, danh sách mô hình đều lấy từ `src/config.py`. **Không tự khai báo lại** trong file của mình. Cần thêm hằng số mới: báo TV5.

| Hằng số hay dùng | Ý nghĩa |
|---|---|
| `ARTICLES_CSV`, `SAMPLE_CSV` | Dữ liệu chính thức, dữ liệu mẫu |
| `EMB_DIR`, `MODELS_DIR`, `TABLES_DIR`, `FIG_DIR` | Thư mục embedding, mô hình, bảng, hình |
| `RESULTS_CSV`, `DOC_TOPICS_CSV`, `TOPIC_WORDS_CSV`, `EMB_TIME_CSV` | Các bảng kết quả (§5) |
| `SEED`, `SEEDS` | `42`; `[42, 43, 44]` khi chạy 3 seed |
| `TOPICS` | 6 khóa chủ đề (`the_thao`, …) → tên hiển thị có dấu |
| `EMBEDDINGS` | 6 khóa mô hình `b0_tfidf_svd` … `b5_e5`, kèm tên mô hình và cột văn bản đầu vào |
| `VECTORIZER_PARAMS`, `BERTOPIC_VECTORIZER_PARAMS`, `UMAP_PARAMS` | Tham số dùng chung |
| `BEST_EMBEDDING`, `BEST_SVD_DIM`, `BEST_MIN_CLUSTER_SIZE` | Cấu hình chốt. Hiện là **giá trị tạm**, TV4 cập nhật sau ngày 4 |

---

## 3. Định dạng dữ liệu & giá trị dùng chung

### 3.1. `data/processed/articles.csv`

| Cột | Kiểu | Ví dụ | Ghi chú |
|---|---|---|---|
| `id` | str | `doc_0001` | Đánh số liên tục sau khi xáo trộn, không lộ nhãn |
| `title` | str | | |
| `sapo` | str | | Đoạn tóm tắt đầu bài |
| `content` | str | | Thân bài, đã bỏ chú thích ảnh, tên tác giả |
| `url` | str | | Duy nhất |
| `source` | str | `vnexpress` | Một giá trị trong `SOURCES` |
| `date` | str | `2026-11-20` | ISO `YYYY-MM-DD` |
| `true_label` | str | `the_thao` | **Một khóa trong `TOPICS`** (không dấu) |
| `text_tok` | str | `bóng_đá việt_nam thắng ...` | Toàn văn, chữ thường, tách từ, bỏ stopwords, **các từ cách nhau bởi 1 dấu cách** |
| `text_bert` | str | `Tuyển Việt Nam thắng...` | Tiêu đề + `". "` + sapo, làm sạch nhẹ |
| `text_bert_seg` | str | `Tuyển Việt_Nam thắng...` | `text_bert` đã tách từ |

- Mã hóa UTF-8, `index=False` khi lưu.
- **Thứ tự dòng là cố định** sau khi đóng băng. Mọi ma trận embedding, nhãn dự đoán đều theo đúng thứ tự này.

### 3.2. Giá trị chuẩn cho `method`, `embedding`, `mode`

| Cột | Giá trị được phép |
|---|---|
| `method` | `lda`, `lsa_kmeans`, `bert_kmeans`, `bertopic` |
| `embedding` | `bow` (cho LDA), `tfidf_svd` (cho LSA), hoặc một khóa trong `EMBEDDINGS` (`b0_tfidf_svd` … `b5_e5`) |
| `mode` | `k6` (số chủ đề cố định = 6), `hdbscan` |
| `param` | Chuỗi mô tả tham số: `svd_dim=10`, `min_cluster_size=30`, hoặc rỗng |

### 3.3. Nhãn dự đoán

- Kiểu `np.ndarray` số nguyên, độ dài bằng số bài, cùng thứ tự với `articles.csv`.
- Chủ đề đánh số từ `0`. Văn bản nhiễu của HDBSCAN có nhãn `NOISE_LABEL = -1`.

---

## 4. Hàm theo từng file

### 4.1. `src/preprocess.py` – TV1

| Hàm | Đầu vào | Trả về | Mô tả |
|---|---|---|---|
| `load_stopwords(path=STOPWORDS_TXT)` | | `set[str]` | Mỗi dòng một từ, dạng đã tách từ (`cho_biết`) |
| `clean_text(text: str)` | | `str` | Bỏ HTML, URL, email, ký tự thừa, chuẩn hóa Unicode NFC, thống nhất cách viết giữa các báo (`TP.HCM`→`TP_HCM`, `tỉ`→`tỷ`, `ôtô`→`ô tô`, `ê-kíp`→`êkíp`), gộp khoảng trắng. **Giữ hoa/thường** |
| `segment(text: str)` | | `str` | Tách từ bằng `pyvi.ViTokenizer.tokenize` |
| `make_text_tok(text: str, stopwords: set[str])` | Văn bản thô | `str` | `clean_text` → `segment` (tách từ khi còn hoa/thường) → chữ thường → bỏ stopwords, dấu câu, số, token 1 ký tự |
| `make_text_bert(title: str, sapo: str)` | | `str` | `clean_text(title + ". " + sapo)` |
| `build_articles(raw_dir=DATA_RAW, seed=SEED)` | | `pd.DataFrame` | Đọc `*.jsonl`, lọc (bỏ trang không phải bài báo), loại trùng url/tiêu đề và gần trùng nội dung (cosine TF-IDF > 0.95), cân bằng `ARTICLES_PER_TOPIC`, tạo 3 cột văn bản, xáo trộn, đánh `id`. Cột đúng `ARTICLE_COLUMNS` |
| `load_articles(sample: bool = False)` | | `pd.DataFrame` | Đọc `SAMPLE_CSV` hoặc `ARTICLES_CSV`. **Mọi người dùng hàm này để đọc dữ liệu** |
| `get_tokens(df: pd.DataFrame)` | | `list[list[str]]` | `df["text_tok"].str.split().tolist()`, dùng cho LDA và coherence |

Chạy: `python -m src.preprocess` → tạo `ARTICLES_CSV`.

### 4.2. `src/crawl.py` – TV1

| Hàm | Đầu vào | Trả về | Mô tả |
|---|---|---|---|
| `list_article_urls(source: str, topic: str, max_pages: int = 10)` | | `list[str]` | Duyệt trang chuyên mục 1 → `max_pages`, lấy link bài |
| `parse_article(url: str, source: str, topic: str)` | | `dict \| None` | Khóa: `title, sapo, content, url, source, date, true_label`. Trả `None` nếu bài không hợp lệ |
| `crawl_topic(topic: str, max_per_source: int = 120)` | | `None` | Ghi nối tiếp vào `DATA_RAW / f"{topic}.jsonl"`, bỏ qua URL đã có |
| `main()` | | | Chạy toàn bộ `TOPICS` × `SOURCES` |

Chạy: `python -m src.crawl`. Bảng chuyên mục từng báo khai báo trong `crawl.py` dưới tên `CATEGORY_URLS = {topic: {source: url}}`.

### 4.3. `src/metrics.py` – TV2 (xong **ngày 2**, mọi người dùng chung)

| Hàm | Đầu vào | Trả về | Mô tả |
|---|---|---|---|
| `purity(labels_true, labels_pred)` | array | `float` | |
| `topic_diversity(topic_words: list[list[str]], top_n=TOP_N_WORDS)` | | `float` | Số từ khác nhau / (số chủ đề × `top_n`) |
| `coherence_cv(topic_words, texts: list[list[str]])` | | `float` | `gensim CoherenceModel(coherence="c_v")`. Bỏ các từ không có trong từ điển trước khi tính |
| `evaluate(labels_pred, labels_true, topic_words, texts, ignore_noise=True)` | | `dict` | Trả về khóa: `ari, nmi, purity, cv, diversity, n_topics, pct_noise`. Khi `ignore_noise=True`, ARI/NMI/Purity tính trên các văn bản có nhãn ≠ `-1` |
| `make_row(method, embedding, mode, param, seed, scores: dict, time_s: float)` | | `dict` | Một dòng đúng cột của `results.csv` |
| `save_results(rows: list[dict])` | | `None` | Ghi vào `RESULTS_CSV`. Dòng trùng khóa (`method, embedding, mode, param, seed`) thì **ghi đè**, không nhân bản |

- `topic_words` luôn là `list[list[str]]`: mỗi phần tử là danh sách `TOP_N_WORDS` từ của một chủ đề, **không gồm chủ đề nhiễu `-1`**.
- `texts` luôn là kết quả của `get_tokens(df)`.

### 4.4. `src/baselines.py` – TV2

| Hàm | Đầu vào | Trả về | Mô tả |
|---|---|---|---|
| `top_words_by_cluster(text_tok: list[str], labels, top_n=TOP_N_WORDS)` | | `list[list[str]]` | TF-IDF (`VECTORIZER_PARAMS`), lấy trung bình mỗi cụm, chọn top từ. **TV3 dùng lại cho A3** |
| `run_lda(texts: list[list[str]], k=N_TOPICS, seed=SEED)` | | `(labels, topic_words, lda_model, dictionary)` | `Dictionary.filter_extremes(no_below=3, no_above=0.5)`, `passes=LDA_PASSES`. Nhãn = chủ đề có xác suất lớn nhất |
| `run_lsa_kmeans(text_tok: list[str], n_dims: int, k=N_TOPICS, seed=SEED)` | | `(labels, topic_words, X_lsa)` | TF-IDF → `TruncatedSVD` → chuẩn hóa L2 → `KMeans(n_init=10)` |
| `save_lda(lda_model, dictionary, path=LDA_DIR)` | | `None` | Lưu `lda.model` và `dictionary.dict` |
| `load_lda(path=LDA_DIR)` | | `(lda_model, dictionary)` | |
| `predict_lda(text: str, lda_model, dictionary)` | Văn bản thô | `(topic_id: int, prob: float)` | Dùng `make_text_tok` rồi suy ra chủ đề. **App dùng hàm này** |

### 4.5. `src/embed.py` – TV3

| Hàm | Đầu vào | Trả về | Mô tả |
|---|---|---|---|
| `embed_texts(texts: list[str], key: str, batch_size=32)` | | `np.ndarray (n, d)` float32, **đã chuẩn hóa L2** | B1–B5 theo `EMBEDDINGS[key]`. `cls`/`mean`: dùng `transformers`, cắt `MAX_LENGTH`. `sbert`: dùng `SentenceTransformer`, thêm `prefix` nếu có |
| `embed_tfidf_svd(text_tok: list[str], dim=100, seed=SEED)` | | `np.ndarray (n, dim)`, chuẩn hóa L2 | B0 |
| `compute_all(df, keys=None)` | | `pd.DataFrame` | Tính các khóa trong `keys` (mặc định tất cả), lưu `EMB_DIR / f"{key}.npy"`, ghi thời gian vào `EMB_TIME_CSV` (cột `embedding, n_docs, dim, time_s`) |
| `load_embedding(key: str)` | | `np.ndarray` | Đọc `.npy`, **kiểm tra số dòng bằng số bài** trong `articles.csv`, sai thì báo lỗi |
| `embed_query(text: str, key: str)` | Văn bản thô | `np.ndarray (1, d)` | Xử lý giống lúc tính embedding (`make_text_bert`, tách từ nếu `text_col == "text_bert_seg"`, thêm `prefix`). **App dùng hàm này** |
| `run_bert_kmeans(X, k=N_TOPICS, seed=SEED)` | | `labels` | `KMeans(n_init=10)` trên embedding đã chuẩn hóa (A3) |

### 4.6. `src/bertopic_runner.py` – TV4

| Hàm | Đầu vào | Trả về | Mô tả |
|---|---|---|---|
| `build_bertopic(mode: str, seed=SEED, min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE)` | `mode` ∈ `k6`, `hdbscan` | `BERTopic` | UMAP(`UMAP_PARAMS`, `random_state=seed`); `hdbscan`: `HDBSCAN(min_cluster_size, prediction_data=True)`; `k6`: `KMeans(N_TOPICS, random_state=seed)`; `CountVectorizer(**VECTORIZER_PARAMS)`; `embedding_model=None`, `top_n_words=TOP_N_WORDS` |
| `fit_bertopic(docs: list[str], embeddings, mode, seed=SEED, min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE)` | `docs` = cột `text_tok` | `(model, labels)` | `model.fit_transform(docs, embeddings=embeddings)` |
| `get_topic_words(model, top_n=TOP_N_WORDS)` | | `list[list[str]]` | Bỏ chủ đề `-1`, sắp theo số thứ tự chủ đề |
| `reduce_noise(model, docs, labels, embeddings)` | | `labels` | `model.reduce_outliers(..., strategy="embeddings")` |
| `coords_2d(embeddings, seed=SEED)` | | `np.ndarray (n, 2)` | UMAP 2 chiều để vẽ bản đồ (cosine) |
| `save_bertopic(model, path=BERTOPIC_DIR)` | | `None` | `model.save(path, serialization="safetensors", save_ctfidf=True, save_embedding_model=False)` |
| `load_bertopic(path=BERTOPIC_DIR)` | | `BERTopic` | |
| `predict_bertopic(text: str, model, key=BEST_EMBEDDING)` | Văn bản thô | `(topic_id: int, prob: float)` | `model.transform([make_text_tok(text)], embeddings=embed_query(text, key))`. **App dùng hàm này** |

### 4.7. `app/streamlit_app.py` – TV5

- **Chỉ đọc** file trong `outputs/`, `embeddings/`, `models/`. Không huấn luyện lại.
- Gọi `predict_lda`, `predict_bertopic` cho Tab 3.
- Tải mô hình bằng `@st.cache_resource`, đọc bảng bằng `@st.cache_data`.
- Tên chủ đề hiển thị lấy từ `TOPICS` (khóa → tên có dấu).

---

## 5. File đầu ra

### 5.1. Bảng (`outputs/tables/`)

| File | Cột | Người ghi |
|---|---|---|
| `results.csv` | `method, embedding, mode, param, seed, n_topics, pct_noise, ari, nmi, purity, cv, diversity, time_s` | TV2 (`lda`, `lsa_kmeans`), TV3 (`bert_kmeans`), TV4 (`bertopic`). Chỉ ghi qua `save_results` |
| `embedding_time.csv` | `embedding, n_docs, dim, time_s` | TV3 |
| `topic_words.csv` | `method, topic, n_docs, words` (`words`: các từ cách nhau 1 dấu cách) | TV2 (`lda`, `lsa_kmeans`), TV4 (`bertopic_k6`, `bertopic_hdbscan`), chỉ cho **cấu hình chốt** |
| `doc_topics.csv` | `id, true_label, x, y, lda, lsa_kmeans, bert_kmeans, bertopic_k6, bertopic_hdbscan` | TV4 tổng hợp (nhãn của **cấu hình chốt**, seed 42; `x, y` từ `coords_2d` của `BEST_EMBEDDING`) |

Số thực làm tròn 4 chữ số thập phân. `time_s` đo bằng `time.perf_counter()`.

### 5.2. Embedding & mô hình

| File / thư mục | Người tạo |
|---|---|
| `embeddings/b0_tfidf_svd.npy` … `embeddings/b5_e5.npy` | TV3 |
| `models/lda/lda.model`, `models/lda/dictionary.dict` | TV2 |
| `models/bertopic_best/` | TV4 |

### 5.3. Hình (`outputs/figures/`, PNG, `dpi=150`)

| File | Nội dung | Người tạo |
|---|---|---|
| `eda_so_bai_chu_de.png` | Số bài theo chủ đề và nguồn | TV1 |
| `eda_do_dai.png` | Phân bố độ dài `text_bert`, `text_tok` | TV1 |
| `fig1_ban_do_2d.png` | Bản đồ 2D: chủ đề tìm được và nhãn thật, đặt cạnh nhau | TV4 |
| `fig2_tu_khoa.png` | Top 10 từ khóa mỗi chủ đề (cấu hình tốt nhất) | TV4 |
| `fig3_heatmap.png` | Chủ đề tìm được × nhãn thật | TV4 |
| `fig4_bang1.png` | Biểu đồ cột nhóm cho Bảng 1 | TV4 |
| `fig5_nmi_thoi_gian.png` | NMI theo thời gian embedding (Bảng 2) | TV4 |

---

## 6. Notebook

| Notebook | Người | Gọi hàm | Ghi ra |
|---|---|---|---|
| `notebooks/01_du_lieu.ipynb` | TV1 | `load_articles` | `eda_*.png` |
| `notebooks/02_mo_hinh_truyen_thong.ipynb` | TV2 | `get_tokens`, `run_lda`, `run_lsa_kmeans` (quét `SVD_DIMS`, 3 seed), `evaluate`, `save_results`, `save_lda` | `results.csv`, `topic_words.csv`, `models/lda/` |
| `notebooks/03_bieu_dien_bert.ipynb` | TV3 | `compute_all`, `load_embedding`, `run_bert_kmeans`, `top_words_by_cluster`, `evaluate`, `save_results` | `embeddings/*.npy`, `embedding_time.csv`, `results.csv` |
| `notebooks/04_bertopic_so_sanh.ipynb` | TV4 | `fit_bertopic` (6 embedding × 2 chế độ × `MIN_CLUSTER_SIZES`, 3 seed), `reduce_noise`, `evaluate`, `save_results`, `coords_2d`, `save_bertopic` | `results.csv`, `topic_words.csv`, `doc_topics.csv`, `fig1`–`fig5`, `models/bertopic_best/` |

Phân tích lỗi (TV2) dùng `doc_topics.csv`, viết ở cuối notebook 02.

---

## 7. Trình tự phụ thuộc

```
config.py (đã có)
   └─ preprocess.py + articles_sample.csv (TV1, ngày 1)
         ├─ metrics.py (TV2, ngày 2) ─────────────────────────────┐
         ├─ baselines.py (TV2) ── top_words_by_cluster ──┐         │
         └─ articles.csv (TV1, ngày 2)                   │         │
               └─ embed.py → embeddings/*.npy (TV3, ngày 3) ──┐   │
                                                              ▼   ▼
                                         bertopic_runner.py (TV4) → results.csv, doc_topics.csv
                                                              │
                                                              ▼
                                                   app/streamlit_app.py (TV5)
```

Trước khi có dữ liệu thật, mọi người phát triển trên `load_articles(sample=True)`.
