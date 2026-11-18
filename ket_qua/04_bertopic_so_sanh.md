# Kết quả – Phần 4: BERTopic & so sánh tổng hợp

- **Người phụ trách:** TV4
- **Code:** `src/bertopic_runner.py`, `notebooks/04_bertopic_so_sanh.ipynb`
- **Đầu ra:** `outputs/tables/results.csv`, `outputs/tables/doc_topics.csv`, `models/bertopic_best/`, `outputs/figures/`

## 1. Tóm tắt

| Hạng mục | Kết quả |
|---|---|
| Embedding tốt nhất (NMI, k = 6) | |
| `min_cluster_size` chọn | |
| Cấu hình tốt nhất tổng thể | |

## 2. Chọn `min_cluster_size`

| Embedding | `min_cluster_size` | Số chủ đề | Tỉ lệ nhiễu | C_v |
|---|---|---|---|---|
| | 15 | | | |
| | 30 | | | |
| | 50 | | | |

## 3. Bảng 1 – Hướng A (k = 6, trung bình ± độ lệch chuẩn 3 seed)

| Phương pháp | ARI | NMI | Purity | C_v | Diversity | Thời gian (s) |
|---|---|---|---|---|---|---|
| A1. LDA | | | | | | |
| A2. LSA + K-Means | | | | | | |
| A3. BERT + K-Means | | | | | | |
| A4. BERTopic | | | | | | |

## 4. Bảng 2 – Hướng B

| Mô hình | NMI (k=6) | ARI (k=6) | NMI (HDBSCAN) | C_v | Diversity | Số chủ đề | Tỉ lệ nhiễu | Thời gian embedding (s) |
|---|---|---|---|---|---|---|---|---|
| B0. TF-IDF + SVD | | | | | | | | |
| B1. mBERT `[CLS]` | | | | | | | | |
| B2. mBERT mean pooling | | | | | | | | |
| B3. Multilingual MiniLM | | | | | | | | |
| B4. Vietnamese-SBERT | | | | | | | | |
| B5. Multilingual-E5 | | | | | | | | |

## 5. Xử lý cụm nhiễu (cấu hình tốt nhất)

| Cách tính | ARI | NMI |
|---|---|---|
| Bỏ văn bản nhiễu | | |
| Sau `reduce_outliers()` | | |

## 6. Chủ đề tìm được (cấu hình tốt nhất)

| Chủ đề | Số bài | Top 10 từ khóa | Nhãn thật chiếm đa số |
|---|---|---|---|
| | | | |

## 7. Hình

| Hình | File |
|---|---|
| Bản đồ 2D (chủ đề tìm được / nhãn thật) | |
| Top từ khóa mỗi chủ đề | |
| Heatmap chủ đề × nhãn thật | |
| Biểu đồ cột Bảng 1 | |
| NMI theo thời gian embedding (Bảng 2) | |

## 8. Thí nghiệm bổ sung (nếu thực hiện)

| Thí nghiệm | Kết quả |
|---|---|
| P1. Độ dài đầu vào | |
| P2. Tập con 20 bài/chủ đề | |

## 9. Nhận xét

- 

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

