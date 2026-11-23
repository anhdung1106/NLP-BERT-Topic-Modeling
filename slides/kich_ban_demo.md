# Kịch bản demo

- **Thời lượng:** ~3 phút
- **Người thao tác:** TV10 · **Người chuẩn bị:** TV5
- **Mô hình dùng trong demo:** BERTopic k = 6 + E5 (`models/bertopic_best/`) và LDA (`models/lda/`)

---

## 0. Chuẩn bị

**Trước 1 ngày** (trên máy sẽ trình chiếu)
- [ ] `git pull`, cài `requirements.txt` (Python 3.12).
- [ ] Chạy `streamlit run app/streamlit_app.py`, vào Tab 3 bấm "Phân tích" một lần để **tải sẵn mô hình E5** (~1 GB). Lần sau không cần mạng cho bước này.
- [ ] Chạy thử toàn bộ kịch bản, đối chiếu cột "Kết quả mong đợi".
- [ ] Chụp màn hình 4 tab làm dự phòng.

**Trước 15 phút**
- [ ] Mở app, chờ Tab 1 hiện bản đồ.
- [ ] Tắt thông báo hệ thống, phóng to trình duyệt (`Cmd/Ctrl +`).

---

## 1. Bản đồ chủ đề – Tab 1 (~45 giây)

| Thao tác | Lời dẫn | Kết quả mong đợi |
|---|---|---|
| Để mô hình **BERTopic (k = 6)** | "Mỗi chấm là một bài báo. BERTopic tự gom thành 6 chủ đề." | 6 cụm màu, chú thích có từ khóa từng chủ đề |
| Bật **Tô màu theo nhãn thật** | "Đối chiếu với chuyên mục thật của báo: Thể thao, Sức khỏe, Giải trí tách rõ; Thời sự, Kinh tế, Công nghệ nằm sát nhau." | Màu chuyên mục gần trùng với cụm chủ đề |

## 2. Chữ "đường" trong hai ngữ cảnh – Tab 3 (~1 phút 30)

Bấm từng câu mẫu, rồi bấm **Phân tích**.

| Câu mẫu | LDA (mong đợi) | BERTopic (mong đợi) |
|---|---|---|
| Giá đường tăng mạnh do nguồn cung khan hiếm | Kinh tế | Kinh tế |
| Đường cao tốc Bắc - Nam đoạn mới chính thức thông xe | **Kinh tế** | **Thời sự** |

**Lời dẫn:** "LDA chỉ đếm từ, thấy chữ 'đường' như nhau nên xếp cả hai câu vào Kinh tế. BERTopic dùng embedding theo ngữ cảnh nên hiểu câu thứ hai nói về giao thông và xếp vào Thời sự."

> Đọc **tên chuyên mục trong ngoặc**, không đọc số chủ đề: số của LDA và BERTopic không tương ứng nhau.

Nếu còn thời gian: mời người nghe đọc một tiêu đề báo hôm nay để nhập thử.

## 3. So sánh mô hình – Tab 4 (~45 giây)

| Thao tác | Lời dẫn |
|---|---|
| Chọn chỉ số **ARI** | "BERTopic k = 6 có ARI cao nhất, 0,62." |
| Chọn chỉ số **NMI** | "Theo NMI thì BERTopic (0,647) ngang LSA (0,651), cả hai đều vượt LDA (0,50)." |
| Chỉ vào cột **Thời gian** | "Đổi lại, BERTopic tốn khoảng 133 giây, chủ yếu cho bước embedding, còn LSA chỉ 2 giây." |

---

## Xử lý sự cố

| Sự cố | Cách xử lý |
|---|---|
| App không mở được | Dùng ảnh chụp màn hình dự phòng |
| Bấm "Phân tích" chờ rất lâu | Mô hình E5 chưa được tải sẵn: dùng ảnh chụp Tab 3, chuyển sang Tab 4 |
| Kết quả câu nhập thêm không như ý | Quay lại 2 câu "đường" đã kiểm chứng |
