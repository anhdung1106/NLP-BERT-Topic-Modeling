# NLP – Kiến trúc BERT và ứng dụng trong Topic Modeling

Seminar môn Xử lý ngôn ngữ tự nhiên: so sánh topic modeling dựa trên BERT (BERTopic) với các phương pháp truyền thống (LDA, LSA + K-Means), và so sánh các cách lấy biểu diễn văn bản từ BERT, trên ~1.800 bài báo tiếng Việt thuộc 6 chuyên mục.

- Kế hoạch chi tiết: [KE_HOACH.md](KE_HOACH.md)
- Phân công: [PHAN_CONG.md](PHAN_CONG.md)
- Hướng dẫn dùng GitHub (đọc trước khi bắt đầu): [HUONG_DAN_GITHUB.md](HUONG_DAN_GITHUB.md)
- Quy ước code (tên hàm, tên file, tên cột): [QUY_UOC_CODE.md](QUY_UOC_CODE.md)

## Cài đặt

Yêu cầu Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy

```bash
streamlit run app/streamlit_app.py     # demo, dùng kết quả và mô hình có sẵn trong repo
```

Chạy lại thực nghiệm (tùy chọn): mở lần lượt `notebooks/01` → `04` và chạy toàn bộ. Embedding đã được tính sẵn trong `embeddings/`, không cần tính lại.

> Lần đầu mở demo, ứng dụng tải mô hình BERT từ Hugging Face (~0,5–1 GB). Nên mở thử một lần trước buổi thuyết trình.

## Cấu trúc thư mục

```
data/        raw/ (dữ liệu crawl, không commit), processed/articles.csv, stopwords/
src/         code dùng chung: crawl, tiền xử lý, embedding, mô hình, chỉ số đánh giá
notebooks/   chạy thực nghiệm và xem kết quả (01 dữ liệu → 04 BERTopic & so sánh)
embeddings/  embedding tính sẵn (<model>.npy)
models/      mô hình dùng cho demo
outputs/     tables/ (bảng kết quả), figures/ (hình)
app/         ứng dụng demo Streamlit
ket_qua/     tóm tắt kết quả + dàn ý slide của từng phần code
slides/      dàn ý lý thuyết, bố cục slide, kịch bản demo, script thuyết trình
```

## Quy ước làm việc

- Mỗi thành viên làm trên branch riêng, merge vào `main` qua Pull Request. Chi tiết: [HUONG_DAN_GITHUB.md](HUONG_DAN_GITHUB.md).
- Không sửa `data/processed/articles.csv` sau khi đã đóng băng (cuối ngày 2).
- Tên hàm, hằng số, cột dữ liệu, file đầu ra: tuân theo [QUY_UOC_CODE.md](QUY_UOC_CODE.md).
