# Kết quả – Phần 5: Ứng dụng demo

- **Người phụ trách:** TV5
- **Code:** `app/streamlit_app.py`
- **Chạy:** `streamlit run app/streamlit_app.py` (từ thư mục gốc repo, đã cài `requirements.txt`)
- **Kịch bản demo:** [../slides/kich_ban_demo.md](../slides/kich_ban_demo.md)

## 1. Chức năng

| Tab | Chức năng | Trạng thái |
|---|---|---|
| 1. Bản đồ chủ đề | Mỗi chấm là một bài báo (UMAP 2 chiều của E5). Chọn 1 trong 5 mô hình; bật/tắt tô màu theo chuyên mục thật; rê chuột xem tiêu đề | ✅ |
| 2. Khám phá chủ đề | Chọn mô hình và chủ đề → số bài, từ khóa, các chuyên mục có trong chủ đề, 5 bài gần tâm chủ đề nhất | ✅ |
| 3. Phân tích văn bản mới | Nhập một đoạn tin (hoặc bấm câu mẫu) → chủ đề của LDA và BERTopic hiện cạnh nhau | ✅ |
| 4. So sánh mô hình | Bảng 1, Bảng 2 kèm biểu đồ; chọn chỉ số để xem | ✅ Số liệu khớp `ket_qua/04` |

App **chỉ đọc kết quả có sẵn**, không huấn luyện lại. Tên chủ đề hiển thị dạng *"Chủ đề t: 3 từ khóa (≈ chuyên mục chiếm đa số)"*.

## 2. Dữ liệu và mô hình app sử dụng

| Thành phần | File | Của |
|---|---|---|
| Bài báo | `data/processed/articles.csv` | TV1 |
| Nhãn chủ đề, tọa độ bản đồ | `outputs/tables/doc_topics.csv` | TV2, TV3, TV4 |
| Từ khóa chủ đề | `outputs/tables/topic_words.csv` | TV2, TV4 |
| Bảng 1, Bảng 2 | `outputs/tables/results.csv`, `embedding_time.csv` | TV2, TV3, TV4 |
| Dự đoán LDA | `models/lda/` + `predict_lda` | TV2 |
| Dự đoán BERTopic (k = 6, E5) | `models/bertopic_best/` + `predict_bertopic` | TV4 |
| Embedding câu mới | `embed_query` (tải mô hình E5 ~1 GB ở lần chạy đầu) | TV3 |

## 3. Kiểm thử

| Hệ điều hành | Python | Kết quả | Ghi chú |
|---|---|---|---|
| macOS | 3.12 | ✅ Cả 4 tab chạy được | Lần đầu bấm "Phân tích" mất vài phút để tải E5, các lần sau vài giây |
| Windows | — | Chưa thử | |

## 4. Kết quả thử Tab 3

| Câu nhập | LDA | BERTopic |
|---|---|---|
| Giá **đường** tăng mạnh do nguồn cung khan hiếm | Kinh tế | Kinh tế |
| **Đường** cao tốc Bắc - Nam đoạn mới chính thức thông xe | **Kinh tế** (sai) | **Thời sự** (đúng) |
| Bác sĩ cảnh báo bệnh sốt xuất huyết tăng mạnh | Sức khỏe | Sức khỏe |

**Nhận xét:** LDA chỉ đếm từ nên thấy chữ "đường" giống nhau, xếp cả hai câu vào cùng một chủ đề. BERTopic dùng embedding theo ngữ cảnh nên tách được hai nghĩa: đường (thực phẩm) và đường (giao thông).

**Lưu ý khi đọc kết quả:** số thứ tự chủ đề của LDA và BERTopic **không tương ứng nhau** (mỗi mô hình tự đánh số). So sánh bằng chuyên mục trong ngoặc, không so bằng số.

## Dàn ý slide

### Slide 1 – Demo: chữ "đường" trong hai ngữ cảnh

- **Ý chính:**
  - Nhập 2 câu có chữ "đường" với nghĩa khác nhau.
  - LDA (đếm từ) xếp cả hai vào Kinh tế; BERTopic (E5) tách đúng Kinh tế và Thời sự.
  - Minh họa trực tiếp ý "vector theo ngữ cảnh" của BERT ở phần lý thuyết.
- **Bảng / hình:** bảng mục 4 + ảnh chụp Tab 3 (dự phòng khi demo lỗi)
- **Điểm nhấn:** đây là ví dụ mở đầu của bài, giờ được chứng minh bằng mô hình thật.
