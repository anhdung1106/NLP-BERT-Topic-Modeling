# Kế hoạch seminar: Kiến trúc BERT và ứng dụng trong Topic Modeling văn bản tin tức tiếng Việt

## 1. Tổng quan

| Hạng mục | Nội dung |
|---|---|
| **Đề tài** | Kiến trúc BERT và ứng dụng trong topic modeling |
| **Câu hỏi nghiên cứu** | **(A)** Topic modeling dựa trên BERT có vượt trội các phương pháp truyền thống (LDA, LSA) hay không, vượt trội ở khía cạnh nào và với chi phí tính toán bao nhiêu? **(B)** Cách lấy biểu diễn từ BERT (BERT nguyên bản hay mô hình đã huấn luyện cho câu; vector `[CLS]` hay trung bình token) ảnh hưởng thế nào đến chất lượng chủ đề? |
| **Loại bài toán** | Học không giám sát. Nhãn chuyên mục của báo **chỉ dùng để đánh giá**, không đưa vào huấn luyện |
| **Dữ liệu** | ~1.800 bài báo tiếng Việt thuộc 6 chuyên mục, thu thập từ 4 báo điện tử |
| **Sản phẩm bàn giao** | Repo GitHub chạy được trên máy cá nhân (CPU), bảng so sánh mô hình, web demo Streamlit, slide thuyết trình |
| **Nhân sự** | 10 thành viên: 5 lập trình (TV1–TV5), 3 thiết kế slide (TV6–TV8), 2 thuyết trình (TV9–TV10) |
| **Thời gian** | Chuẩn bị 7 ngày, thuyết trình **15 phút** |

**Định hướng nội dung:** lý thuyết chỉ trình bày những gì cần để hiểu phần thực nghiệm. **Trọng tâm là kết quả thực nghiệm.** Phân bổ dự kiến khoảng 30% lý thuyết và 70% thực nghiệm, demo.

### Phân biệt với các nhóm khác

| Nhóm | Đề tài | Cách xử lý phần giao nhau |
|---|---|---|
| 1 | LSTM và word segmentation | Tách từ chỉ là bước tiền xử lý, không trình bày |
| 2 | Transformer và text classification | Kiến trúc Transformer chỉ nhắc lại trong 1 slide. Đề tài này là học không giám sát, không huấn luyện bộ phân loại |
| 4 | PhoBERT và gán nhãn từ loại | PhoBERT chỉ giới thiệu trong 1 câu, là một trong các mô hình được so sánh. Không lấy PhoBERT làm trọng tâm |

---

## 2. Nội dung lý thuyết

Dàn ý chi tiết: [slides/dan_y_ly_thuyet.md](slides/dan_y_ly_thuyet.md).

| Khối | Nội dung cần trình bày | Thời lượng | Liên kết với thực nghiệm |
|---|---|---|---|
| **T1. Topic modeling & giới hạn của bag-of-words** | Mục tiêu của topic modeling. LDA (văn bản là hỗn hợp chủ đề) và LSA (phân rã SVD). Giới hạn chung: biểu diễn dựa trên tần suất từ, không nắm được ngữ cảnh và từ đồng nghĩa, phải chọn trước số chủ đề | 1–2 slide | Hướng A: A1, A2 |
| **T2. Nhắc lại Transformer encoder** | Self-attention cho phép mỗi token tổng hợp thông tin từ toàn câu. Chi tiết đã được nhóm 2 trình bày | 1 slide | Nền tảng cho T3 |
| **T3. Kiến trúc BERT** | Chỉ dùng phần encoder, xếp chồng nhiều lớp (base: 12 lớp, 768 chiều, ~110M tham số). Đọc **hai chiều**. Biểu diễn đầu vào = token embedding (WordPiece) + segment embedding + position embedding. Token đặc biệt `[CLS]`, `[SEP]`. Độ dài tối đa 512 token. Tiền huấn luyện bằng **MLM** (che 15% token rồi dự đoán) và **NSP**. Đầu ra: một vector **theo ngữ cảnh** cho mỗi token. Hai cách sử dụng: fine-tuning và **trích xuất đặc trưng** (cách dùng trong đề tài này). Biến thể: mBERT (104 ngôn ngữ), PhoBERT (tiếng Việt) | 3–4 slide | Hướng B: B1, B2 |
| **T4. Từ vector token đến vector văn bản** | Hai cách gộp: lấy vector `[CLS]` hoặc trung bình các token (mean pooling). BERT nguyên bản không được huấn luyện cho việc so sánh câu, nên các vector này chưa phản ánh tốt độ tương đồng ngữ nghĩa. **Sentence-BERT** fine-tune BERT theo kiến trúc siamese để câu cùng nghĩa có vector gần nhau | 1–2 slide | Hướng B: so sánh B1, B2 với B3–B5 |
| **T5. BERTopic** | Pipeline 4 bước: Embedding → UMAP (giảm chiều) → HDBSCAN (phân cụm theo mật độ, tự xác định số cụm, có cụm nhiễu -1) → c-TF-IDF (trích từ khóa đại diện cho mỗi cụm) | 1–2 slide | Toàn bộ thực nghiệm và demo |

**Sơ đồ đối chiếu dùng trong slide:**

```
Truyền thống:  Văn bản → BoW/TF-IDF → LSA (SVD) → K-Means → Từ khóa
BERTopic:      Văn bản → BERT/SBERT → UMAP      → HDBSCAN → c-TF-IDF
                         ▲ biểu diễn theo ngữ cảnh         ▲ tự xác định số chủ đề
```

---

## 3. Dữ liệu

### 3.1. Nguồn & phương pháp thu thập

Thu thập từ trang chuyên mục của 4 báo, duyệt qua nhiều trang của mỗi chuyên mục (trang 1, 2, 3...).

| Chủ đề | VnExpress | Tuổi Trẻ | Thanh Niên | Dân trí |
|---|---|---|---|---|
| Thời sự | thoi-su | thoi-su | thoi-su | xa-hoi |
| Kinh tế | kinh-doanh | kinh-doanh | kinh-te | kinh-doanh |
| Thể thao | the-thao | the-thao | the-thao | the-thao |
| Giải trí | giai-tri | giai-tri | giai-tri | giai-tri |
| Công nghệ | khoa-hoc-cong-nghe | cong-nghe | cong-nghe | cong-nghe |
| Sức khỏe | suc-khoe | suc-khoe | suc-khoe | suc-khoe |

**Quy trình:**

```
Với mỗi chủ đề × mỗi báo
│
├─ [CRAWL]  Duyệt trang chuyên mục 1 → 2 → 3 ... → thu thập link bài (lọc theo mẫu URL của từng báo)
│           → dừng khi đủ ~120 link ứng viên hoặc hết trang
│
└─ Với mỗi link
     ├─ Đã tải trước đó → bỏ qua (cho phép chạy tiếp sau khi gián đoạn)
     ├─ [SCRAPE] Trích tiêu đề, sapo, thân bài, ngày đăng; đối chiếu breadcrumb để xác nhận chuyên mục
     ├─ Loại bài video/ảnh, bài dưới 150 từ
     └─ [LƯU] Ghi nối tiếp vào data/raw/<chu_de>.jsonl
│
└─ [GỘP] Loại trùng theo URL và tiêu đề → cân bằng 300 bài/chủ đề → data/processed/articles.csv
```

- Khoảng cách giữa các request **~1 giây**, gửi tuần tự. Lỗi mạng hoặc HTTP 429/5xx: chờ tăng dần rồi thử lại, tối đa 3 lần.
- **Ngày 1 chạy thử** 2 trang mỗi báo để kiểm tra cơ chế phân trang. Báo nào phân trang bằng JavaScript thì loại khỏi danh sách và bù số lượng từ các báo còn lại.
- `true_label` lấy theo chuyên mục gốc. `id` đánh lại dạng `doc_0001` để không lộ nhãn.

### 3.2. Quy mô

| Hạng mục | Mục tiêu |
|---|---|
| Số bài | **300 bài/chủ đề × 6 = ~1.800 bài**. Ngưỡng tối thiểu: 200 bài/chủ đề |
| Cân bằng | Chênh lệch số bài giữa các chủ đề ≤ 10% |
| Thời gian thu thập | ~2.500 request, khoảng 45–60 phút |

Quy mô này cần thiết vì HDBSCAN phân cụm dựa trên mật độ điểm. Với dữ liệu quá nhỏ, phần lớn văn bản sẽ bị xếp vào cụm nhiễu và kết quả thiếu tin cậy.

### 3.3. Các phiên bản văn bản

| Cột | Nội dung | Sử dụng cho |
|---|---|---|
| `text_bert` | Tiêu đề + sapo, làm sạch nhẹ | Các mô hình BERT. Độ dài ngắn, nằm trong giới hạn token |
| `text_bert_seg` | Như `text_bert`, đã tách từ | Mô hình nền PhoBERT (B4) |
| `text_tok` | Toàn văn, chữ thường, tách từ, loại stopwords | LDA, LSA, trích từ khóa c-TF-IDF, tính coherence |

---

## 4. Tiền xử lý

| Bước | `text_tok` | `text_bert` | `text_bert_seg` |
|---|---|---|---|
| Loại HTML, URL, ký tự thừa | ✅ | ✅ | ✅ |
| Chuẩn hóa Unicode NFC | ✅ | ✅ | ✅ |
| Chuyển chữ thường | ✅ | ❌ | ❌ |
| Tách từ (`pyvi`) | ✅ | ❌ | ✅ |
| Loại stopwords | ✅ | ❌ | ❌ |
| Loại từ xuất hiện trong < 3 bài hoặc > 50% số bài | ✅ (trong vectorizer) | — | — |

**Nguyên tắc:** đầu vào của BERT được giữ ở dạng câu tự nhiên để mô hình khai thác ngữ cảnh. Stopwords chỉ loại ở bước trích từ khóa hiển thị, để danh sách từ khóa có nghĩa.

Danh sách stopwords: `data/stopwords/vietnamese_stopwords.txt`.

---

## 5. Pipeline BERTopic

### 5.1. Cấu hình mặc định

| Bước | Thư viện | Cấu hình | Ghi chú |
|---|---|---|---|
| Embedding | `transformers`, `sentence-transformers` | Thay đổi theo Hướng B (§6.2) | Tính một lần, lưu `.npy`, commit lên repo |
| Giảm chiều | `umap-learn` | `n_neighbors=15, n_components=5, metric='cosine', random_state=42` | Cố định seed để tái lập kết quả |
| Phân cụm | `hdbscan` | `min_cluster_size` ∈ {15, 30, 50} | Tham số chính quyết định số chủ đề |
| Trích từ khóa | `CountVectorizer` + c-TF-IDF | Trên `text_tok` (đã tách từ, từ ghép như `bóng_đá` là một token) | Dùng unigram để tính coherence nhất quán giữa các mô hình |

### 5.2. Hai chế độ chạy

| Chế độ | Phân cụm | Mục đích |
|---|---|---|
| **Tự do** | HDBSCAN | Đúng thiết kế gốc: tự xác định số chủ đề, có cụm nhiễu |
| **Cố định k = 6** | K-Means thay cho HDBSCAN | So sánh công bằng với LDA/LSA (cùng số chủ đề, không có nhiễu) |

### 5.3. Xử lý cụm nhiễu khi đánh giá

1. Tính ARI/NMI trên các văn bản không thuộc cụm nhiễu, kèm **tỉ lệ nhiễu**.
2. Gán lại văn bản nhiễu vào chủ đề gần nhất bằng `reduce_outliers()`, rồi tính lại trên toàn bộ dữ liệu.

---

## 6. Thiết kế thực nghiệm

### 6.1. Hướng A: BERT so với phương pháp truyền thống (k = 6)

| # | Phương pháp | Biểu diễn | Phân cụm | Vai trò |
|---|---|---|---|---|
| A1 | **LDA** | BoW | `gensim LdaModel` | Mô hình chủ đề xác suất kinh điển |
| A2 | **LSA + K-Means** | TF-IDF → SVD (10/20/50 chiều) | K-Means | Mô hình truyền thống dạng vector |
| A3 | **BERT + K-Means** | Embedding tốt nhất ở Hướng B | K-Means | Chỉ thay biểu diễn, giữ nguyên thuật toán phân cụm |
| A4 | **BERTopic** | Embedding tốt nhất ở Hướng B | UMAP + HDBSCAN | Pipeline đầy đủ |

**Phân tích đóng góp:** A2 → A3 đo riêng tác động của **biểu diễn BERT**. A3 → A4 đo tác động của **UMAP + HDBSCAN**.

### 6.2. Hướng B: cách lấy biểu diễn từ BERT

| # | Mô hình | Cách lấy vector văn bản | Vai trò |
|---|---|---|---|
| B0 | TF-IDF + SVD (100 chiều) | — | Mốc đối chứng không dùng BERT |
| B1 | `bert-base-multilingual-cased` (mBERT) | Vector **`[CLS]`** | BERT nguyên bản, chưa huấn luyện cho tác vụ câu |
| B2 | `bert-base-multilingual-cased` (mBERT) | **Mean pooling** | Cùng mô hình với B1, chỉ khác cách gộp |
| B3 | `paraphrase-multilingual-MiniLM-L12-v2` | Mean pooling (SBERT) | SBERT đa ngôn ngữ, kích thước nhỏ |
| B4 | `keepitreal/vietnamese-sbert` | Mean pooling (SBERT) | SBERT trên nền PhoBERT, chuyên cho tiếng Việt (cần tách từ) |
| B5 | `intfloat/multilingual-e5-base` | Mean pooling | Mô hình embedding đa ngôn ngữ thế hệ mới (tiền tố `"query: "`) |

**Các phép so sánh trọng tâm, gắn với lý thuyết:**

| So sánh | Kiểm chứng nội dung lý thuyết |
|---|---|
| B1 với B2 | Vector `[CLS]` hay mean pooling phù hợp hơn khi dùng BERT nguyên bản (T4) |
| B1, B2 với B3–B5 | BERT nguyên bản so với mô hình đã huấn luyện cho tác vụ câu (T4, Sentence-BERT) |
| B3 với B4 | Mô hình đa ngôn ngữ so với mô hình chuyên tiếng Việt |
| B0 với các mô hình còn lại | Biểu diễn theo ngữ cảnh so với biểu diễn dựa trên tần suất (T1) |

Mỗi mô hình chạy ở **cả hai chế độ**. Mô hình có NMI cao nhất ở chế độ k = 6 được dùng cho A3, A4 và làm mặc định trong demo.

### 6.3. Thí nghiệm bổ sung (thực hiện nếu hoàn thành Hướng A, B trước ngày 4)

| # | Thí nghiệm | Câu hỏi |
|---|---|---|
| P1 | Đầu vào: tiêu đề / tiêu đề + sapo / tiêu đề + sapo + đoạn đầu | Lượng văn bản đưa vào BERT ảnh hưởng thế nào đến chất lượng chủ đề? |
| P2 | Chạy trên tập con 20 bài/chủ đề | BERTopic có còn hiệu quả khi dữ liệu ít không? |

### 6.4. Nguyên tắc so sánh

- Tất cả mô hình dùng **cùng dữ liệu, cùng số văn bản**.
- **Cố định seed.** Cấu hình tốt nhất của mỗi phương pháp chạy 3 seed, báo cáo trung bình ± độ lệch chuẩn.
- Siêu tham số chọn theo **chỉ số nội tại** (coherence, silhouette). Nhãn thật chỉ dùng cho đánh giá cuối cùng.

---

## 7. Đánh giá

### 7.1. Bộ chỉ số

| Nhóm | Chỉ số | Ý nghĩa | Công cụ |
|---|---|---|---|
| **Mức khớp với nhãn thật** | ARI, NMI, Purity | Mức độ trùng khớp giữa cụm tìm được và chuyên mục gốc | `scikit-learn`, hàm Purity tự cài đặt |
| **Chất lượng chủ đề** | Coherence C_v | Mức độ các từ khóa trong cùng một chủ đề thường xuất hiện cùng nhau | `gensim CoherenceModel` |
| | Topic diversity | Tỉ lệ từ không trùng lặp trong top 10 từ khóa của tất cả chủ đề | Tự cài đặt |
| **Chi phí** | Thời gian embedding, thời gian huấn luyện (CPU) | Chi phí tính toán tương ứng với chất lượng | `time.perf_counter` |
| **Đặc thù BERTopic** | Số chủ đề tìm được, tỉ lệ nhiễu | Khả năng tự xác định số chủ đề của HDBSCAN | |

### 7.2. Bảng kết quả

**Bảng 1: Hướng A (k = 6)**

| Phương pháp | ARI | NMI | Purity | C_v | Diversity | Thời gian (s) |
|---|---|---|---|---|---|---|
| A1. LDA | | | | | | |
| A2. LSA + K-Means | | | | | | |
| A3. BERT + K-Means | | | | | | |
| A4. BERTopic | | | | | | |

**Bảng 2: Hướng B**

| Mô hình | NMI (k=6) | ARI (k=6) | NMI (HDBSCAN) | C_v | Diversity | Số chủ đề | Tỉ lệ nhiễu | Thời gian embedding (s) |
|---|---|---|---|---|---|---|---|---|
| B0. TF-IDF + SVD | | | | | | | | |
| B1. mBERT `[CLS]` | | | | | | | | |
| B2. mBERT mean pooling | | | | | | | | |
| B3. Multilingual MiniLM (SBERT) | | | | | | | | |
| B4. Vietnamese-SBERT | | | | | | | | |
| B5. Multilingual-E5 | | | | | | | | |

### 7.3. Hình minh họa (`outputs/figures/`)

1. **Bản đồ 2D** các văn bản, tô màu theo chủ đề tìm được và theo nhãn thật, đặt cạnh nhau.
2. **Biểu đồ cột** top 10 từ khóa của mỗi chủ đề.
3. **Heatmap** chủ đề tìm được × nhãn thật.
4. **Biểu đồ cột nhóm** cho Bảng 1.
5. **Biểu đồ phân tán** NMI theo thời gian embedding cho Bảng 2, thể hiện quan hệ chất lượng và chi phí.

### 7.4. Phân tích lỗi

- Chọn 5–10 văn bản LDA phân sai nhưng BERTopic phân đúng (và ngược lại), phân tích nguyên nhân.
- Xác định các cặp chủ đề thường bị nhầm lẫn và lý giải.
- Đánh giá hiện tượng HDBSCAN tách một chuyên mục thành nhiều chủ đề con (ví dụ: thể thao thành bóng đá và các môn khác).

---

## 8. Demo: ứng dụng Streamlit

Toàn bộ mô hình và kết quả được **tính trước và lưu trong repo**. Ứng dụng chỉ đọc lên, không phải huấn luyện lại khi chạy. Kịch bản chi tiết: [slides/kich_ban_demo.md](slides/kich_ban_demo.md).

| Tab | Chức năng | Tương tác |
|---|---|---|
| **1. Bản đồ chủ đề** | Biểu diễn 2D các văn bản, tô màu theo chủ đề | Hiện tiêu đề khi rê chuột; chọn mô hình; chuyển sang tô màu theo nhãn thật |
| **2. Khám phá chủ đề** | Danh sách chủ đề, từ khóa, văn bản tiêu biểu | Chọn chủ đề để xem chi tiết |
| **3. Phân tích văn bản mới** | Nhập một đoạn tin → chủ đề dự đoán và xác suất | Kết quả LDA và BERTopic hiển thị song song; có sẵn các câu mẫu đa nghĩa |
| **4. So sánh mô hình** | Bảng 1, Bảng 2 và biểu đồ | Chọn chỉ số cần hiển thị |

**Phương án dự phòng:** video quay sẵn quá trình demo (~2 phút).

---

## 9. Tổ chức mã nguồn

### 9.1. Script hay notebook

| Loại | Dùng cho | Lý do |
|---|---|---|
| **`src/*.py`** | Code dùng chung: thu thập, tiền xử lý, embedding, mô hình, chỉ số đánh giá | Notebook và app Streamlit cùng import, tránh chép code; dễ review qua Pull Request |
| **`notebooks/*.ipynb`** | Chạy thực nghiệm, in bảng, vẽ hình | Lưu kèm output nên xem được kết quả trực tiếp trên GitHub; thuận tiện khi trình bày kết quả |

Notebook chỉ gọi hàm trong `src/`, không chứa logic dài.

### 9.2. Cấu trúc thư mục

```
NLP-BERT-Topic-Modeling/
├── README.md                       # cài đặt, cách chạy, cấu trúc
├── KE_HOACH.md                     # file này
├── PHAN_CONG.md                    # phân công
├── QUY_UOC_CODE.md                 # tên hàm, hằng số, cột dữ liệu, file đầu ra
├── HUONG_DAN_GITHUB.md             # hướng dẫn dùng GitHub cho thành viên
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                        # <chu_de>.jsonl (không commit)
│   ├── processed/articles.csv      # dữ liệu chính thức (commit)
│   └── stopwords/vietnamese_stopwords.txt
├── embeddings/                     # <model>.npy (commit, ~5 MB/mô hình)
├── models/                         # BERTopic (safetensors), LDA dùng cho demo
├── outputs/
│   ├── tables/                     # results.csv, doc_topics.csv, topic_words.csv, embedding_time.csv
│   └── figures/
│
├── src/
│   ├── config.py                   # đường dẫn, seed, danh sách mô hình, tham số chung
│   ├── crawl.py                    # thu thập dữ liệu
│   ├── preprocess.py               # tạo text_tok / text_bert / text_bert_seg
│   ├── embed.py                    # trích xuất & lưu embedding B0–B5
│   ├── baselines.py                # LDA, LSA + K-Means
│   ├── bertopic_runner.py          # BERTopic (HDBSCAN / K-Means)
│   └── metrics.py                  # ARI, NMI, Purity, C_v, diversity
├── notebooks/
│   ├── 01_du_lieu.ipynb            # thống kê mô tả dữ liệu
│   ├── 02_mo_hinh_truyen_thong.ipynb  # A1, A2
│   ├── 03_bieu_dien_bert.ipynb     # tính embedding B0–B5, A3
│   └── 04_bertopic_so_sanh.ipynb   # Hướng B, A4, 3 seed, bảng & hình, phân tích lỗi
├── app/streamlit_app.py
│
├── ket_qua/                        # tóm tắt kết quả + dàn ý slide từng phần
│   ├── 01_du_lieu.md
│   ├── 02_mo_hinh_truyen_thong.md
│   ├── 03_bieu_dien_bert.md
│   ├── 04_bertopic_so_sanh.md
│   └── 05_demo.md
└── slides/
    ├── dan_y_ly_thuyet.md
    ├── bo_cuc_slide.md
    ├── kich_ban_demo.md
    └── script_thuyet_trinh.md
```

Tên hàm, hằng số, cột dữ liệu, file đầu ra của từng phần: xem [QUY_UOC_CODE.md](QUY_UOC_CODE.md).

---

## 10. Phân công (tóm tắt)

Chi tiết: [PHAN_CONG.md](PHAN_CONG.md).

| Thành viên | Vai trò | Phạm vi chính |
|---|---|---|
| TV1 | Lập trình | Dữ liệu: thu thập, tiền xử lý, thống kê mô tả |
| TV2 | Lập trình | Mô hình truyền thống (A1, A2), bộ chỉ số đánh giá, phân tích lỗi |
| TV3 | Lập trình | Biểu diễn BERT (B0–B5), A3 |
| TV4 | Lập trình | BERTopic, Hướng B, A4, bảng và hình kết quả |
| TV5 | Lập trình | Quản lý repo, duyệt và merge PR, ứng dụng demo |
| TV6 | Slide | Template, slide lý thuyết |
| TV7 | Slide | Slide dữ liệu và phương pháp |
| TV8 | Slide | Slide kết quả, demo, kết luận |
| TV9 | Thuyết trình | Mở đầu, lý thuyết, dữ liệu & phương pháp |
| TV10 | Thuyết trình | Kết quả, demo, kết luận |

---

## 11. Tiến độ 7 ngày

| Ngày | Lập trình | Slide | Thuyết trình | Mốc kiểm soát |
|---|---|---|---|---|
| **1** | TV5 mời thành viên vào repo, phổ biến quy ước. TV1 chạy thử thu thập, cung cấp `articles_sample.csv`. TV2, TV3, TV4 phát triển trên dữ liệu mẫu | TV6 thiết kế template | — | Repo hoạt động, có dữ liệu mẫu |
| **2** | TV1 thu thập đầy đủ và tiền xử lý. TV2: LDA, LSA, `metrics.py`. TV3: `embed.py`. TV4: `bertopic_runner.py`. TV5: Tab 1, 2 | TV6 slide lý thuyết; TV7 slide dữ liệu | — | **Đóng băng dữ liệu** cuối ngày; dàn ý lý thuyết hoàn tất |
| **3** | TV3 tính và commit toàn bộ embedding, chạy A3. TV2 chạy A1, A2. TV4 chạy Hướng B, quét tham số. TV5: Tab 3 | TV7 slide phương pháp | TV9, TV10 nắm lý thuyết, cài đặt repo, chạy thử demo | Đủ embedding trên repo |
| **4** | TV4 chạy A4 và 3 seed. TV2 phân tích lỗi. Thí nghiệm bổ sung nếu kịp | TV8 dựng khung slide kết quả | — | **Đóng băng kết quả** |
| **5** | TV4 xuất hình. TV5 hoàn thiện Tab 4, kiểm thử trên máy khác. TV1–TV5 hoàn thành `ket_qua/` | TV7, TV8 đưa bảng và hình vào slide | TV9, TV10 soạn script | Demo hoàn chỉnh, đủ dàn ý |
| **6** | Sửa lỗi; quay video dự phòng | Hoàn thiện slide | Hoàn thiện script; **tổng duyệt lần 1** có bấm giờ | Slide gần hoàn chỉnh |
| **7** | Chỉ sửa lỗi nghiêm trọng | Chỉnh theo góp ý tổng duyệt | **Tổng duyệt lần 2** trên máy trình chiếu; luyện hỏi đáp | **Sẵn sàng thuyết trình** |

**Phương án khi chậm tiến độ:**
- Dữ liệu chưa đủ cuối ngày 2: chấp nhận 200 bài/chủ đề và đóng băng.
- Ngày 4 chưa hoàn thành: bỏ B5 và thí nghiệm bổ sung.
- Demo chậm: giữ Tab 1, 3, 4.
- **Không cắt giảm:** Bảng 1, Bảng 2, các phép so sánh trọng tâm ở §6.2 và Tab 3 của demo.

### 11.1. Ước tính thời gian chạy (CPU, ~1.800 văn bản, đầu vào tiêu đề + sapo)

| Tác vụ | Ước tính |
|---|---|
| Thu thập dữ liệu (~2.500 request) | 45–60 phút |
| Tiền xử lý | 1–3 phút |
| Embedding B1, B2 (mBERT, dùng chung một lượt chạy) | 3–6 phút |
| Embedding B3 (MiniLM) | 1–2 phút |
| Embedding B4 (Vietnamese-SBERT) | 3–6 phút |
| Embedding B5 (E5-base) | 4–8 phút |
| LDA (1 lần) | 1–2 phút |
| BERTopic (1 cấu hình, đã có embedding) | 10–30 giây |
| Toàn bộ notebook 02–04 (đã có embedding) | 20–40 phút |
| Tải mô hình lần đầu khi mở demo | 0,5–1 GB; cần chạy demo trước buổi thuyết trình để lưu cache |

*Các con số trên là ước tính, cần cập nhật sau lần chạy đầu tiên.*

---

## 12. Cấu trúc bài thuyết trình (15 phút)

Chi tiết: [slides/bo_cuc_slide.md](slides/bo_cuc_slide.md).

| Phần | Nội dung | Số slide | Thời lượng | Trình bày |
|---|---|---|---|---|
| Mở đầu | Bài toán topic modeling, câu hỏi nghiên cứu | 1 | 0,5' | TV9 |
| Lý thuyết | T1 → T5 | 7–9 | 4' | TV9 |
| Dữ liệu & phương pháp | Dữ liệu, Hướng A và B, bộ chỉ số | 2 | 1,5' | TV9 |
| Kết quả | Bảng 1, Bảng 2, hình minh họa, phân tích lỗi | 4–5 | 5' | TV10 |
| Demo | Trực tiếp trên Streamlit | — | 3' | TV10 |
| Kết luận | Tổng kết, khuyến nghị sử dụng, hạn chế | 1 | 1' | TV10 |
| **Tổng** | | **15–18** | **15'** | |

---

## 13. Rủi ro & phương án xử lý

| Rủi ro | Phương án |
|---|---|
| Không thu thập đủ dữ liệu (thay đổi giao diện, phân trang động) | Kiểm tra phân trang ngày 1; thay thế nguồn; ngưỡng tối thiểu 200 bài/chủ đề |
| Bị chặn khi thu thập | Request tuần tự ~1 giây, thử lại có chờ, hỗ trợ chạy tiếp |
| Các phần phụ thuộc nhau gây chậm | Chốt quy ước file ngày 1, dùng dữ liệu mẫu để phát triển song song |
| Lỗi cài đặt thư viện (đặc biệt `hdbscan` trên Windows) | Hướng dẫn cài đặt trong README; kiểm thử đa nền tảng ngày 5; dùng máy đã kiểm thử để trình chiếu |
| Máy cấu hình thấp | Embedding được commit sẵn, không cần tính lại |
| Kết quả không ổn định giữa các lần chạy | Cố định `random_state`; báo cáo 3 seed |
| HDBSCAN cho tỉ lệ nhiễu cao hoặc quá nhiều chủ đề nhỏ | Quét `min_cluster_size`; `reduce_outliers()`; luôn có chế độ k = 6 để đối chiếu |
| BERTopic không vượt LSA về ARI | Trình bày như một kết quả thực nghiệm có giá trị: văn bản tin tức có từ khóa đặc trưng rõ nên phương pháp tần suất đã đạt hiệu quả cao; bổ sung so sánh ở coherence, diversity và câu đa nghĩa |
| Mô hình phải tải lần đầu khi demo, mạng chậm | Chạy demo trước buổi thuyết trình để lưu cache; video dự phòng |
| Vượt thời lượng 15 phút | Hai lần tổng duyệt có bấm giờ; giới hạn lý thuyết trong 4 phút |
| `ket_qua/` hoàn thành trễ | Hạn cứng cuối ngày 5; nhóm slide dựng sẵn khung từ kế hoạch này |

---

## 14. Tài liệu tham khảo

- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
- Nguyen, D. Q., & Nguyen, A. T. (2020). *PhoBERT: Pre-trained language models for Vietnamese.* Findings of EMNLP.
- Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure.* arXiv:2203.05794.
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). *Latent Dirichlet Allocation.* JMLR.
- Deerwester, S. et al. (1990). *Indexing by Latent Semantic Analysis.* JASIS.
- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection.* arXiv:1802.03426.
- Campello, R., Moulavi, D., & Sander, J. (2013). *Density-Based Clustering Based on Hierarchical Density Estimates.* PAKDD.
- Röder, M., Both, A., & Hinneburg, A. (2015). *Exploring the Space of Topic Coherence Measures.* WSDM.
- BERTopic Documentation: https://maartengr.github.io/BERTopic/
