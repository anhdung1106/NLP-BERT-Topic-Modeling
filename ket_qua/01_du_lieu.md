# Kết quả – Phần 1: Dữ liệu

- **Người phụ trách:** TV1 (Phúc)
- **Code:** `src/crawl.py`, `src/preprocess.py`, `notebooks/01_du_lieu.ipynb`
- **Đầu ra:** `data/processed/articles.csv`, hình thống kê trong `outputs/figures/`

## 1. Tóm tắt

| Tiêu chí | Mục tiêu | Kết quả |
|---|---|---|
| Tổng số bài | ~1.800 | 1.800 |
| Số chủ đề | 6 | 6 |
| Số bài mỗi chủ đề | 300 (tối thiểu 200) | 300 (cân bằng tuyệt đối) |
| Số nguồn báo | 4 | 4 |
| Ngày thu thập | | 20/11/2026 |

## 2. Nguồn sử dụng

| Báo | Phân trang hoạt động | Ghi chú |
|---|---|---|
| VnExpress | Có | Tìm được ~315 link mỗi chủ đề, đủ 120 bài mỗi chủ đề. Đóng góp 657 bài (36,5%) |
| Tuổi Trẻ | Có, nhưng ít link | Chỉ tìm được ~16–21 link mới mỗi chủ đề (phân trang không trả thêm nhiều link). Đóng góp 147 bài (8,2%) |
| Thanh Niên | Có | Tìm được ~65 link mỗi chủ đề, giữ 47–63 bài mỗi chủ đề. Đóng góp 335 bài (18,6%) |
| Dân trí | Có | Tìm được ~515 link mỗi chủ đề, đủ 120 bài mỗi chủ đề. Đóng góp 661 bài (36,7%) |

Chuyên mục dùng cho từng chủ đề: xem `CATEGORY_URLS` trong `src/crawl.py` (ví dụ Thời sự = `thoi-su` ở VnExpress, Tuổi Trẻ, Thanh Niên và `xa-hoi` ở Dân trí).

## 3. Kết quả thu thập

Từ **1.927** bài thô, sau khi lọc (bỏ bài dưới 150 từ, tiêu đề quá ngắn) và loại trùng còn **1.914** bài, rồi cân bằng về **1.800** bài.

| Chủ đề | Sau lọc và loại trùng | Giữ lại (cân bằng) | Bị loại khi cân bằng |
|---|---|---|---|
| Thời sự | 322 | 300 | 22 |
| Kinh tế | 306 | 300 | 6 |
| Thể thao | 324 | 300 | 24 |
| Giải trí | 313 | 300 | 13 |
| Công nghệ | 326 | 300 | 26 |
| Sức khỏe | 323 | 300 | 23 |
| **Tổng** | **1.914** | **1.800** | **114** |

Khi cân bằng, mỗi chủ đề lấy lần lượt từng báo một bài cho đến đủ 300, nên báo ít bài (Tuổi Trẻ) được giữ gần như toàn bộ.

Số bài theo chủ đề × nguồn (sau cân bằng):

| Chủ đề | VnExpress | Tuổi Trẻ | Thanh Niên | Dân trí | Tổng |
|---|---|---|---|---|---|
| Thời sự | 108 | 20 | 63 | 109 | 300 |
| Kinh tế | 116 | 15 | 52 | 117 | 300 |
| Thể thao | 107 | 38 | 47 | 108 | 300 |
| Giải trí | 111 | 18 | 59 | 112 | 300 |
| Công nghệ | 107 | 34 | 52 | 107 | 300 |
| Sức khỏe | 108 | 22 | 62 | 108 | 300 |
| **Tổng** | **657** | **147** | **335** | **661** | **1.800** |

## 4. Thống kê mô tả

| Chỉ số | `text_bert` | `text_tok` |
|---|---|---|
| Số từ trung bình | 49,6 | 364,3 |
| Số từ nhỏ nhất / lớn nhất | 12 / 122 | 87 / 2.093 |
| Kích thước từ vựng | — | 28.386 |

- `text_bert` (tiêu đề + sapo): trung vị 49 từ, dùng cho các mô hình BERT.
- `text_tok` (toàn văn đã chữ thường, tách từ, bỏ stopwords): trung vị 321 từ, tổng cộng 655.721 token, dùng cho LDA, LSA và rút từ khóa.
- Toàn văn gốc (`content`): trung bình 786 từ, trung vị 683 từ.

Số từ trung bình theo chủ đề:

| Chủ đề | `content` | `text_bert` | `text_tok` |
|---|---|---|---|
| Thời sự | 752,8 | 51,5 | 353,0 |
| Kinh tế | 823,5 | 50,5 | 378,9 |
| Thể thao | 707,0 | 48,3 | 329,8 |
| Giải trí | 885,2 | 48,4 | 396,4 |
| Công nghệ | 799,7 | 46,9 | 363,1 |
| Sức khỏe | 748,4 | 51,8 | 364,5 |

Hình: `outputs/figures/eda_so_bai_chu_de.png`, `outputs/figures/eda_do_dai.png`

## 5. Nhận xét

- **Cân bằng chủ đề, không cân bằng nguồn.** Mỗi chủ đề đúng 300 bài, nhưng Dân trí (36,7%) và VnExpress (36,5%) chiếm phần lớn, còn Tuổi Trẻ chỉ 8,2%. Nhãn là chuyên mục chứ không phải báo, nên điều này ít ảnh hưởng đến đánh giá, nhưng là một hạn chế cần nêu.
- **Dữ liệu thiên về tin gần đây.** 82% bài đăng trong 12 ngày cuối (09–20/11/2026), vì trang chuyên mục chỉ hiển thị tin mới. Có một số bài cũ hơn (159 bài đăng trước 01/09/2026).
- **Nhãn lấy theo chuyên mục gốc của báo, nên có chồng lấn tự nhiên.** Ví dụ bài về Anthropic nằm ở cả Công nghệ và Kinh tế; Thời sự lấy từ chuyên mục "Xã hội" của Dân trí. Đây là giới hạn khi dùng ARI, NMI để đánh giá.
- **Văn bản đầu vào BERT ngắn** (trung bình ~50 từ, tối đa 122 từ), phần lớn nằm trong giới hạn `MAX_LENGTH = 256` token, nên đa số bài không bị cắt.
- **Chất lượng sau làm sạch tốt, còn sót rất ít chữ thừa** (khoảng 26 dòng trên 1.800 bài: chú thích "Đồ họa:", dòng "Xem màn trình diễn ... trực tiếp" của Dân trí, dòng "Địa chỉ:" trong bài quảng bá). Có 1 bài không có sapo (khi đó `text_bert` chỉ gồm tiêu đề) và 1 trang không phải bài báo thật ("Chính sách bảo mật dữ liệu cá nhân" của Dân trí, mục Công nghệ). Cả hai không đáng kể so với quy mô dữ liệu.
- **Tách từ bằng `pyvi`**: tách trước rồi mới chuyển chữ thường để nhận ra tên riêng (`việt_nam` xuất hiện 3.384 lần, tách lẻ `việt` còn 349 lần).

## Dàn ý slide

> Mỗi slide: tiêu đề, 2–4 ý chính, bảng/hình sử dụng, điểm cần nhấn mạnh khi thuyết trình.

### Slide 1 – Dữ liệu: 1.800 bài báo tiếng Việt, 6 chủ đề

- **Ý chính:**
  - 6 chuyên mục: Thời sự, Kinh tế, Thể thao, Giải trí, Công nghệ, Sức khỏe.
  - Thu thập từ 4 báo (VnExpress, Tuổi Trẻ, Thanh Niên, Dân trí) bằng `requests` + BeautifulSoup.
  - Cân bằng 300 bài mỗi chủ đề; lọc bài trùng, bài dưới 150 từ, bài video/ảnh.
  - Nhãn chuyên mục gốc (`true_label`) chỉ dùng để đánh giá, không đưa vào mô hình.
- **Bảng / hình:** `outputs/figures/eda_so_bai_chu_de.png`
- **Điểm nhấn:** bài toán không giám sát; nhãn chỉ để chấm điểm. Nêu hạn chế: Tuổi Trẻ chỉ chiếm 8% dữ liệu.

### Slide 2 – Tiền xử lý: 3 phiên bản văn bản cho 3 mục đích

- **Ý chính:**
  - `text_bert`: tiêu đề + sapo, giữ nguyên câu và stopwords → các mô hình BERT.
  - `text_bert_seg`: như trên, đã tách từ → Vietnamese-SBERT (nền PhoBERT).
  - `text_tok`: toàn văn, chữ thường, tách từ, bỏ stopwords → LDA, LSA, từ khóa, coherence.
  - Độ dài: `text_bert` ~50 từ, `text_tok` ~360 từ, từ vựng 28.386 từ.
- **Bảng / hình:** bảng 3 phiên bản (có ví dụ cùng một bài); `outputs/figures/eda_do_dai.png`
- **Điểm nhấn:** BERT không bỏ stopwords để hiểu ngữ cảnh; tách từ bằng `pyvi` giúp giữ từ ghép như `việt_nam`, `bóng_đá`.
