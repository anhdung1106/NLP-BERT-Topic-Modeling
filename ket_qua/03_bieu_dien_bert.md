# Kết quả – Phần 3: Biểu diễn văn bản từ BERT

- **Người phụ trách:** TV3
- **Code:** `src/embed.py`, `notebooks/03_bieu_dien_bert.ipynb`
- **Đầu ra:** `embeddings/*.npy`, `outputs/tables/embedding_time.csv`, dòng A3 trong `outputs/tables/results.csv`

## 1. Embedding đã tính

| # | Mô hình | Cách gộp | Đầu vào | Số chiều | Thời gian (s, CPU) | File |
|---|---|---|---|---|---|---|
| B0 | TF-IDF + SVD | — | `text_tok` | 100 | | `b0_tfidf_svd.npy` |
| B1 | mBERT | `[CLS]` | `text_bert` | 768 | | `b1_mbert_cls.npy` |
| B2 | mBERT | Mean pooling | `text_bert` | 768 | | `b2_mbert_mean.npy` |
| B3 | Multilingual MiniLM | Mean pooling | `text_bert` | 384 | | `b3_minilm.npy` |
| B4 | Vietnamese-SBERT | Mean pooling | `text_bert_seg` | 768 | | `b4_vi_sbert.npy` |
| B5 | Multilingual-E5 | Mean pooling | `"query: "` + `text_bert` | 768 | | `b5_e5.npy` |

Cấu hình máy chạy: _CPU, RAM_

## 2. BERT + K-Means (A3, k = 6)

| Embedding | ARI | NMI | Purity | Silhouette |
|---|---|---|---|---|
| B0 | | | | |
| B1 | | | | |
| B2 | | | | |
| B3 | | | | |
| B4 | | | | |
| B5 | | | | |

## 3. Nhận xét

| So sánh | Kết quả | Kết luận |
|---|---|---|
| B1 với B2 (`[CLS]` và mean pooling) | | |
| B1, B2 với B3–B5 (BERT nguyên bản và SBERT) | | |
| B3 với B4 (đa ngôn ngữ và tiếng Việt) | | |
| B0 với các mô hình BERT | | |

## Dàn ý slide

> Mỗi slide: tiêu đề, 2–4 ý chính, bảng/hình sử dụng, điểm cần nhấn mạnh khi thuyết trình.

### Slide 1 – _tiêu đề_

- **Ý chính:**
  - 
- **Bảng / hình:** `outputs/figures/...`
- **Điểm nhấn:** 

### Slide 2 – _tiêu đề_

- **Ý chính:**
  - 
- **Bảng / hình:** 
- **Điểm nhấn:** 

