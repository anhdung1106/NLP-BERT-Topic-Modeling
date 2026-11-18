# Dàn ý lý thuyết

- **Thời lượng:** ~4 phút · 7–9 slide
- **Nguyên tắc:** chỉ trình bày những gì cần để hiểu phần thực nghiệm. Mỗi khối ghi rõ phần thực nghiệm nào dùng đến.
- **Hạn hoàn tất:** cuối ngày 2

---

## T1. Topic modeling & giới hạn của bag-of-words (1–2 slide)

- **Dùng ở:** Hướng A (A1, A2)
- **Ý cần trình bày:**
  - Topic modeling là gì, dùng để làm gì
  - LDA: 
  - LSA: 
  - Giới hạn: 
- **Hình / ví dụ:** 

## T2. Nhắc lại Transformer encoder (1 slide)

- **Dùng ở:** nền tảng cho T3
- **Ý cần trình bày:**
  - Self-attention: 
  - Ghi chú: chi tiết do nhóm 2 trình bày
- **Hình / ví dụ:** 

## T3. Kiến trúc BERT (3–4 slide)

- **Dùng ở:** Hướng B (B1, B2)
- **Ý cần trình bày:**
  - Encoder-only, xếp chồng nhiều lớp; đọc hai chiều
  - Biểu diễn đầu vào: token + segment + position embedding; `[CLS]`, `[SEP]`
  - Tiền huấn luyện: MLM, NSP
  - Đầu ra: vector theo ngữ cảnh cho mỗi token
  - Hai cách sử dụng: fine-tuning và trích xuất đặc trưng
  - Biến thể: mBERT, PhoBERT (1 câu)
- **Hình / ví dụ:** 

## T4. Từ vector token đến vector văn bản (1–2 slide)

- **Dùng ở:** Hướng B (B1, B2 so với B3–B5)
- **Ý cần trình bày:**
  - `[CLS]` và mean pooling
  - Vì sao BERT nguyên bản chưa phù hợp để so sánh câu
  - Sentence-BERT: 
- **Hình / ví dụ:** 

## T5. BERTopic (1–2 slide)

- **Dùng ở:** toàn bộ thực nghiệm và demo
- **Ý cần trình bày:**
  - Embedding → UMAP → HDBSCAN → c-TF-IDF
  - Đối chiếu với pipeline truyền thống
- **Hình / ví dụ:** sơ đồ đối chiếu ở KE_HOACH §2

---

## Ghi chú

- 
