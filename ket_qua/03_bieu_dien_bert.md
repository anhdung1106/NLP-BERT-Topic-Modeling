# Kết quả – Phần 3: Biểu diễn văn bản từ BERT

- **Người phụ trách:** Quốc Phi
- **Code:** `src/embed.py`, `notebooks/03_bieu_dien_bert.ipynb`
- **Đầu ra:** `embeddings/*.npy`, `outputs/tables/embedding_time.csv`, dòng A3 trong `outputs/tables/results.csv`, cột `bert_kmeans` trong `outputs/tables/doc_topics.csv`

## 1. Embedding đã tính

| # | Mô hình | Cách gộp | Đầu vào | Số chiều | Thời gian (s, CPU) | File |
|---|---|---|---|---|---|---|
| B0 | TF-IDF + SVD | — | `text_tok` | 100 | 1,1 | `b0_tfidf_svd.npy` |
| B1 | mBERT | `[CLS]` | `text_bert` | 768 | 182,7 | `b1_mbert_cls.npy` |
| B2 | mBERT | Mean pooling | `text_bert` | 768 | 173,2 | `b2_mbert_mean.npy` |
| B3 | Multilingual MiniLM | Mean pooling | `text_bert` | 384 | 37,4 | `b3_minilm.npy` |
| B4 | Vietnamese-SBERT | Mean pooling | `text_bert_seg` | 768 | 104,4 | `b4_vi_sbert.npy` |
| B5 | Multilingual-E5 | Mean pooling | `"query: "` + `text_bert` | 768 | 129,1 | `b5_e5.npy` |

Cấu hình máy chạy: CPU 12 luồng, không dùng GPU, _điền tên CPU và RAM_. Mỗi file là ma trận 1.800 × d, float32, đã chuẩn hóa L2, cùng thứ tự dòng với `articles.csv`. Cắt tối đa 256 token. Thời gian không gồm thời gian tải mô hình (đã chạy khởi động trước khi đo). B1 và B2 được tính riêng từng lần nên mỗi khóa đều gánh đủ thời gian chạy mBERT.

## 2. BERT + K-Means (A3, k = 6, seed 42)

| Embedding | ARI | NMI | Purity | C_v | Diversity | Thời gian (s) |
|---|---|---|---|---|---|---|
| B0 (TF-IDF + SVD, đối chứng) | 0,5425 | 0,6646 | 0,7406 | 0,7920 | 0,9500 | 1,4 |
| B1 (mBERT `[CLS]`) | 0,1337 | 0,1652 | 0,3467 | 0,4450 | 0,8500 | 183,1 |
| B2 (mBERT mean) | 0,3742 | 0,4389 | 0,6350 | 0,7114 | 0,9500 | 173,5 |
| B3 (MiniLM) | 0,3848 | 0,4531 | 0,6306 | 0,7709 | 0,8667 | 37,6 |
| B4 (Vietnamese-SBERT) | 0,3149 | 0,4179 | 0,5706 | 0,7173 | 0,9500 | 104,8 |
| B5 (E5) | 0,5226 | 0,5707 | 0,7200 | 0,6864 | 0,9667 | 129,5 |

- Thời gian ở bảng này = thời gian embedding + thời gian K-Means.
- Bảng dùng C_v và Diversity (có trong `results.csv`) thay cho Silhouette, vì bộ chỉ số chung của nhóm (`evaluate` trong `metrics.py`) không tính Silhouette.
- **Embedding BERT tốt nhất: B5 (E5)**, NMI = 0,5707 (seed 42).
- Chạy 3 seed (42, 43, 44) cho B5: NMI = 0,5838 ± 0,0197; ARI = 0,5312 ± 0,0124; Purity = 0,7239 ± 0,0082; C_v = 0,7196 ± 0,0434; Diversity = 0,9611 ± 0,0096.

### Chủ đề tìm được với B5 (seed 42)

| Cụm | Từ khóa đầu | Nhãn thật chiếm nhiều nhất |
|---|---|---|
| 0 | doanh_nghiệp, việt_nam, công_nghệ, đầu_tư, kinh_tế | Kinh tế 136, Thời sự 116, Công nghệ 49 (lẫn) |
| 1 | nvidia, huang, iphone, ceo, apple | Công nghệ 226 |
| 2 | đồng, giá, giảm, triệu, usd, vàng | Thời sự 157, Kinh tế 102 (lẫn) |
| 3 | trận, đội_tuyển, bóng, cầu_thủ, hlv | Thể thao 265 |
| 4 | bác_sĩ, người_bệnh, bệnh, điều_trị, ung_thư | Sức khỏe 257 |
| 5 | diễn_viên, phim, ca_sĩ, nghệ_sĩ, khán_giả | Giải trí 255 |

Thể thao, Sức khỏe, Giải trí và Công nghệ tách rõ. Thời sự và Kinh tế bị trộn vào cụm 0 và cụm 2, vì cùng chia sẻ từ vựng về giá cả, đầu tư, đất đai.

## 3. Nhận xét

| So sánh | Kết quả | Kết luận |
|---|---|---|
| B1 với B2 (`[CLS]` và mean pooling) | NMI 0,1652 và 0,4389 | Mean pooling cao hơn khoảng 2,7 lần. BERT nguyên bản không được huấn luyện để vector `[CLS]` đại diện cho nghĩa cả câu, nên khi không fine-tune thì nên lấy trung bình các token |
| B1, B2 với B3–B5 (BERT nguyên bản và SBERT) | Tốt nhất mBERT: 0,4389. Tốt nhất SBERT (B5): 0,5707 | Mô hình đã huấn luyện cho tác vụ câu cho chủ đề khớp nhãn hơn rõ rệt (hơn khoảng 0,13 NMI). Phù hợp lý thuyết Sentence-BERT. Riêng B3 và B4 chỉ ngang hoặc nhỉnh hơn B2 một chút, nên lợi thế không đến từ mọi mô hình SBERT mà chủ yếu từ B5 |
| B3 với B4 (đa ngôn ngữ và tiếng Việt) | NMI 0,4531 và 0,4179 | Mô hình chuyên tiếng Việt không hơn MiniLM đa ngôn ngữ. Chênh lệch nhỏ và chỉ có một seed, nên không nên khẳng định mô hình nào hơn |
| B0 với các mô hình BERT | NMI 0,6646 và 0,5707 (BERT tốt nhất) | Biểu diễn tần suất TF-IDF + SVD **cao hơn mọi mô hình BERT**, lại nhanh hơn hàng trăm lần (1,4 s so với 129 s). Tin tức có từ khóa đặc trưng rõ nên phương pháp tần suất đã làm tốt. Lưu ý B0 dùng toàn văn, còn BERT chỉ dùng tiêu đề + sapo |

Quan hệ chất lượng và chi phí: B3 (37 s, NMI 0,45) là mô hình BERT rẻ nhất, B5 (129 s, NMI 0,57) tốt nhất. B1 và B2 vừa chậm nhất vừa kém hơn SBERT.

### Hạn chế

- BERT chỉ nhận tiêu đề + sapo (cắt 256 token), còn B0 nhận toàn văn, nên so sánh B0 với BERT chưa hoàn toàn công bằng về lượng thông tin.
- Kết quả theo K-Means (k = 6) với seed 42; chỉ cấu hình tốt nhất (B5) được chạy 3 seed.
- Thời gian đo trên một máy CPU, một lần chạy.

## Dàn ý slide

> Mỗi slide: tiêu đề, 2–4 ý chính, bảng/hình sử dụng, điểm cần nhấn mạnh khi thuyết trình.

### Slide 1 – Sáu cách biểu diễn văn bản (Hướng B)

- **Ý chính:**
  - B0 là mốc đối chứng không dùng BERT (TF-IDF + SVD 100 chiều).
  - B1 và B2 dùng cùng mBERT, chỉ khác cách gộp: `[CLS]` hay mean pooling.
  - B3, B4, B5 là các mô hình Sentence-BERT / embedding (đa ngôn ngữ, tiếng Việt, E5).
  - Mọi vector chuẩn hóa L2, K-Means k = 6 trên 1.800 bài.
- **Bảng / hình:** bảng mục 1 của file này
- **Điểm nhấn:** thay đổi duy nhất giữa các dòng là cách tạo vector; thuật toán phân cụm giữ nguyên.

### Slide 2 – Kết quả: SBERT vượt BERT nguyên bản, nhưng TF-IDF vẫn dẫn đầu

- **Ý chính:**
  - `[CLS]` chỉ đạt NMI 0,17, mean pooling đạt 0,44.
  - Mô hình SBERT tốt nhất (E5) đạt NMI 0,57 (3 seed: 0,58 ± 0,02).
  - TF-IDF + SVD đạt 0,66, cao hơn mọi mô hình BERT.
- **Bảng / hình:** bảng mục 2 của file này (kết quả A3, K-Means)
- **Điểm nhấn:** BERT không tự động tốt hơn; cần đúng loại mô hình và so với đối chứng.

### Slide 3 – Chi phí và chủ đề tìm được

- **Ý chính:**
  - E5 mất khoảng 129 s, MiniLM khoảng 37 s, TF-IDF chỉ 1 s (CPU).
  - Với E5, Thể thao, Sức khỏe, Giải trí, Công nghệ tách rõ.
  - Thời sự và Kinh tế bị trộn do chung từ vựng giá cả, đầu tư.
- **Bảng / hình:** bảng "Chủ đề tìm được với B5"
- **Điểm nhấn:** chất lượng chủ đề phụ thuộc cả vào việc các chuyên mục có chồng lấn từ vựng hay không.
