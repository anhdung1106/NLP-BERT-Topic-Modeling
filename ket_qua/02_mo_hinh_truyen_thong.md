# Kết quả – Phần 2: Mô hình truyền thống & đánh giá

- **Người phụ trách:** TV2
- **Code:** `src/baselines.py`, `src/metrics.py`, `notebooks/02_mo_hinh_truyen_thong.ipynb`
- **Đầu ra:** dòng A1, A2 trong `outputs/tables/results.csv`, `models/lda/`

## 1. Tóm tắt

| Phương pháp | Cấu hình chọn | ARI | NMI | Purity | C_v | Diversity | Thời gian (s) |
|---|---|---|---|---|---|---|---|
| A1. LDA (k = 6) | BoW, passes = 10 | 0,448 ± 0,032 | 0,503 ± 0,022 | 0,652 ± 0,021 | 0,679 ± 0,039 | 0,922 ± 0,025 | 23,6 |
| A2. LSA + K-Means (k = 6) | TF-IDF → SVD 50 chiều | 0,572 ± 0,046 | 0,651 ± 0,015 | 0,786 ± 0,027 | 0,792 ± 0,004 | 0,978 ± 0,010 | 2,0 |


## 2. Bộ chỉ số (`src/metrics.py`)

| Chỉ số | Cách tính | Khoảng giá trị |
|---|---|---|
| ARI | | |
| NMI | | |
| Purity | | |
| C_v | | |
| Topic diversity | | |

## 3. Chọn siêu tham số

| Số chiều SVD | C_v | Silhouette | NMI (tham khảo) |
|---|---|---|---|
| 10 | 0,755 ± 0,033 | 0,391 ± 0,004 | 0,638 |
| 20 | 0,780 ± 0,004 | 0,231 ± 0,001 | 0,676 |
| 50 ✔ | 0,792 ± 0,004 | 0,128 ± 0,001 | 0,651 |

## 4. Chủ đề tìm được (top 10 từ khóa)

| Chủ đề | LDA | LSA + K-Means |
|---|---|---|
| 0 | việt_nam, đội, trận, bóng, đội_tuyển, giải, cầu_thủ, hlv, thi_đấu, thắng (278 bài) | trận, đội, đội_tuyển, bóng, cầu_thủ, hlv, việt_nam, u17, giải, thắng (272 bài) |
| 1 | nvidia, công_ty, huang, ceo, mỹ, lãnh_đạo, vụ, jensen, chip, công_nghệ (159 bài) | nvidia, usd, đồng, huang, giá, cổ_phiếu, tỷ, công_ty, ceo, thị_trường (198 bài) |
| 2 | đồng, giá, tỷ, triệu, giảm, usd, tăng, tiền, thị_trường, đầu_tư (281 bài) | bác_sĩ, điều_trị, người_bệnh, bệnh, bệnh_viện, ung_thư, ăn, thuốc, đau, thận (288 bài) |
| 3 | bác_sĩ, bệnh, dùng, điều_trị, người_bệnh, bệnh_viện, giảm, tăng, tình_trạng, y_tế (398 bài) | doanh_nghiệp, việt_nam, công_nghệ, dự_án, đầu_tư, đất, dân, kinh_tế, tỉnh, nước (589 bài) |
| 4 | việt_nam, doanh_nghiệp, công_nghệ, nước, kinh_tế, đầu_tư, xây_dựng, sản_xuất, sản_phẩm, nghiên_cứu (349 bài) | phim, diễn_viên, khán_giả, ca_sĩ, nghệ_sĩ, văn_hóa, tác_phẩm, chương_trình, vai, âm_nhạc (282 bài) |
| 5 | phim, diễn_viên, chương_trình, văn_hóa, khán_giả, tuổi, gia_đình, ăn, bộ, việt_nam (335 bài) | iphone, dùng, thiết_bị, apple, màn_hình, điện_thoại, pin, camera, tính_năng, samsung (171 bài) |

## 5. Phân tích lỗi (LDA và BERTopic)

| ID | Tiêu đề | Nhãn thật | LDA | BERTopic | Nguyên nhân |
|---|---|---|---|---|---|
| | | | | | |

**Các cặp chủ đề hay nhầm lẫn:**

- 

## 6. Nhận xét

- **LSA + K-Means vượt LDA trên mọi chỉ số** (NMI 0,651 so với 0,503; C_v 0,792 so với 0,679) và chạy nhanh hơn khoảng 12 lần (2,0 s so với 23,6 s). Một lý do có thể: LDA giả định mỗi bài là hỗn hợp nhiều chủ đề, khi ép mỗi bài về một nhãn (chủ đề có xác suất cao nhất) thì các bài lẫn lộn dễ bị gán sai.
- **LDA kém ổn định giữa các seed**: độ lệch chuẩn C_v là 0,039, gấp gần 10 lần LSA 50 chiều (0,004); ARI dao động 0,411–0,469 tùy seed.
- **Ba chuyên mục được nhận ra tốt**: Thể thao, Giải trí, Sức khỏe. Ở cả hai mô hình, 263–282/300 bài của mỗi nhãn này rơi vào cùng một chủ đề, với từ khóa rõ ràng (trận, đội_tuyển, cầu_thủ / phim, diễn_viên, khán_giả / bác_sĩ, điều_trị, bệnh_viện).
- **Nhầm lẫn chủ yếu giữa Thời sự – Kinh tế – Công nghệ**: LSA gộp 285 bài Thời sự, 150 bài Kinh tế và 87 bài Công nghệ vào một cụm 589 bài; LDA trộn ba nhãn này trong cả ba chủ đề 1, 2, 4. Ba chuyên mục này dùng chung nhiều từ (doanh_nghiệp, đầu_tư, việt_nam, dự_án), nên biểu diễn dựa trên tần suất từ khó tách.
- **Cả hai mô hình tách riêng một chủ đề "Nvidia"** (LDA 159 bài, LSA 198 bài; từ khóa nvidia, huang, ceo, chip, cổ_phiếu) dù không có chuyên mục nào như vậy. Dữ liệu được thu thập trong thời gian ngắn nên một sự kiện nóng chiếm nhiều bài; bag-of-words gom theo từ trùng (tên riêng) thay vì theo chuyên mục, kéo cả bài Kinh tế lẫn Công nghệ vào cùng một cụm.
- **LSA tách được Công nghệ thiết bị** (cụm iphone, apple, điện_thoại, samsung: 162/171 bài là Công nghệ), trong khi LDA không có chủ đề Công nghệ riêng.
- **Silhouette giảm khi tăng số chiều** (0,391 → 0,231 → 0,128) vì trong không gian nhiều chiều, khoảng cách giữa các điểm kém phân biệt hơn; chỉ số này không so sánh công bằng được giữa các mức số chiều. Vì vậy chọn số chiều theo C_v: **50 chiều** (C_v 0,792). 20 chiều có NMI cao hơn (0,676 so với 0,651) nhưng không chọn, vì NMI dựa trên nhãn thật, dùng nó để chọn tham số là trái với tinh thần học không giám sát.
- **Hạn chế**: cả hai mô hình chỉ dựa vào tần suất từ, không hiểu nghĩa câu; bài cùng chủ đề nhưng dùng từ khác nhau sẽ bị tách, bài khác chủ đề nhưng dùng chung từ sẽ bị gộp. Đây là điểm BERTopic (dùng embedding ngữ nghĩa) cần cải thiện — so sánh cụ thể ở mục 5.

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

