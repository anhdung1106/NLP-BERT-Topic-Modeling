# Kết quả – Phần 4: BERTopic & so sánh tổng hợp

- **Người phụ trách:** TV4
- **Code:** `src/bertopic_runner.py`, `notebooks/04_bertopic_so_sanh.ipynb`
- **Đầu ra:** dòng `bertopic` trong `outputs/tables/results.csv` (72 dòng: 6 embedding × 4 cấu hình × 3 seed), `bertopic_k6` / `bertopic_hdbscan` trong `topic_words.csv`, cột `x, y, bertopic_k6, bertopic_hdbscan` trong `doc_topics.csv`, `models/bertopic_best/`, `outputs/figures/fig1`–`fig5`

## 1. Tóm tắt

| Hạng mục | Kết quả |
|---|---|
| Embedding tốt nhất (NMI, BERTopic k = 6, TB 3 seed) | **B5. Multilingual-E5** (`b5_e5`), NMI 0,647 ± 0,004. Trùng với kết quả A3 của TV3 nên A3 không cần chạy lại |
| `min_cluster_size` chọn | **30** (C_v cao nhất trên E5: 0,825) |
| Cấu hình tốt nhất tổng thể | **BERTopic k = 6 + E5**: ARI 0,623, NMI 0,647, Purity 0,809 – ARI và Purity cao nhất trong cả Bảng 1. Mô hình này được lưu vào `models/bertopic_best/` cho demo |
| Thời gian | Fit BERTopic chỉ ~4 s (HDBSCAN) đến ~9 s (k = 6) khi đã có embedding; chi phí chủ yếu là embedding (E5: 129 s) |

Cấu hình UMAP: `n_neighbors=15, n_components=5, min_dist=0, metric=cosine`. Từ khóa c-TF-IDF trên `text_tok`. Mọi kết quả là trung bình 3 seed (42, 43, 44) trừ khi ghi rõ.

## 2. Chọn `min_cluster_size`

Chọn theo C_v (chỉ số nội) trên embedding tốt nhất, không nhìn nhãn thật. NMI chỉ để tham khảo.

| Embedding | `min_cluster_size` | Số chủ đề | Tỉ lệ nhiễu | C_v | Diversity | NMI (tham khảo) |
|---|---|---|---|---|---|---|
| B5. Multilingual-E5 | 15 | 18,0 | 20,2% | 0,753 | 0,839 | 0,635 |
| B5. Multilingual-E5 | **30 ✔** | 5,0 | 2,1% | **0,825** | 0,860 | 0,627 |
| B5. Multilingual-E5 | 50 | 5,0 | 4,6% | 0,825 | 0,860 | 0,623 |

<details>
<summary>Các embedding còn lại</summary>

| Embedding | `min_cluster_size` | Số chủ đề | Tỉ lệ nhiễu | C_v | Diversity | NMI (tham khảo) |
|---|---|---|---|---|---|---|
| B0. TF-IDF + SVD | 15 | 33,7 | 18,5% | 0,785 | 0,833 | 0,580 |
| B0. TF-IDF + SVD | 30 | 12,0 | 19,7% | 0,810 | 0,892 | 0,646 |
| B0. TF-IDF + SVD | 50 | 5,0 | 2,8% | 0,794 | 0,980 | 0,657 |
| B1. mBERT [CLS] | 15 | 2,7 | 0,4% | 0,482 | 0,900 | 0,035 |
| B1. mBERT [CLS] | 30 | 2,0 | 0,0% | 0,450 | 0,900 | 0,010 |
| B1. mBERT [CLS] | 50 | 2,0 | 0,0% | 0,450 | 0,900 | 0,010 |
| B2. mBERT mean pooling | 15 | 18,3 | 20,9% | 0,743 | 0,868 | 0,535 |
| B2. mBERT mean pooling | 30 | 9,7 | 16,5% | 0,694 | 0,918 | 0,455 |
| B2. mBERT mean pooling | 50 | 3,3 | 3,9% | 0,680 | 0,983 | 0,341 |
| B3. Multilingual MiniLM | 15 | 24,3 | 17,8% | 0,782 | 0,863 | 0,543 |
| B3. Multilingual MiniLM | 30 | 8,7 | 18,4% | 0,744 | 0,894 | 0,421 |
| B3. Multilingual MiniLM | 50 | 3,7 | 8,7% | 0,691 | 0,967 | 0,225 |
| B4. Vietnamese-SBERT | 15 | 23,3 | 26,1% | 0,773 | 0,873 | 0,589 |
| B4. Vietnamese-SBERT | 30 | 6,7 | 10,4% | 0,773 | 0,910 | 0,424 |
| B4. Vietnamese-SBERT | 50 | 7,3 | 14,3% | 0,794 | 0,912 | 0,554 |

</details>

**Lưu ý:** C_v cao nhất lại rơi vào cấu hình gộp ba chuyên mục thành một cụm lớn (xem mục 6). C_v chấm từng chủ đề riêng lẻ nên không "phạt" việc gộp; chọn tham số chỉ bằng C_v có giới hạn này. Nếu chọn 15, HDBSCAN tìm 18 chủ đề con, NMI 0,635, ARI 0,599, nhưng 20% văn bản bị xếp vào nhiễu.

## 3. Bảng 1 – Hướng A (k = 6, trung bình ± độ lệch chuẩn 3 seed)

A3 và A4 dùng cùng embedding E5. A4 có hai dòng: k = 6 (UMAP + K-Means) để so sánh công bằng với A1–A3, và HDBSCAN (pipeline BERTopic đầy đủ).

| Phương pháp | ARI | NMI | Purity | C_v | Diversity | Thời gian (s) |
|---|---|---|---|---|---|---|
| A1. LDA | 0,448 ± 0,032 | 0,503 ± 0,022 | 0,652 ± 0,021 | 0,679 ± 0,039 | 0,922 ± 0,025 | 23,6 |
| A2. LSA + K-Means | 0,572 ± 0,046 | **0,651 ± 0,015** | 0,786 ± 0,027 | 0,792 ± 0,004 | **0,978 ± 0,010** | **2,0** |
| A3. BERT + K-Means | 0,531 ± 0,012 | 0,584 ± 0,020 | 0,724 ± 0,008 | 0,720 ± 0,043 | 0,961 ± 0,010 | 129,6 |
| A4. BERTopic (k = 6) | **0,623 ± 0,004** | 0,647 ± 0,004 | **0,809 ± 0,003** | 0,711 ± 0,003 | 0,967 ± 0,000 | 133,5 |
| A4. BERTopic (HDBSCAN, 5 chủ đề, 2,1% nhiễu) | 0,432 ± 0,002 | 0,627 ± 0,001 | 0,623 ± 0,002 | **0,825 ± 0,000** | 0,860 ± 0,000 | 133,2 |

Thời gian A3, A4 = thời gian embedding E5 (129,1 s) + thời gian phân cụm.

## 4. Bảng 2 – Hướng B (BERTopic, trung bình 3 seed; HDBSCAN với `min_cluster_size = 30`)

| Mô hình | NMI (k=6) | ARI (k=6) | NMI (HDBSCAN) | C_v (k=6) | C_v (HDBSCAN) | Diversity (k=6) | Số chủ đề (HDBSCAN) | Tỉ lệ nhiễu | Thời gian embedding (s) |
|---|---|---|---|---|---|---|---|---|---|
| B0. TF-IDF + SVD | 0,626 ± 0,003 | 0,460 | 0,647 | 0,820 | 0,810 | 0,978 | 12,0 | 19,7% | 1,1 |
| B1. mBERT `[CLS]` | 0,376 ± 0,001 | 0,285 | 0,010 | 0,661 | 0,450 | 0,828 | 2,0 | 0,0% | 182,7 |
| B2. mBERT mean pooling | 0,506 ± 0,007 | 0,445 | 0,455 | 0,675 | 0,694 | 0,922 | 9,7 | 16,5% | 173,2 |
| B3. Multilingual MiniLM | 0,591 ± 0,005 | 0,549 | 0,421 | 0,719 | 0,744 | 0,939 | 8,7 | 18,4% | 37,4 |
| B4. Vietnamese-SBERT | 0,583 ± 0,007 | 0,540 | 0,424 | 0,759 | 0,773 | 0,967 | 6,7 | 10,4% | 104,4 |
| B5. Multilingual-E5 | **0,647 ± 0,004** | **0,623** | 0,627 | 0,711 | **0,825** | 0,967 | 5,0 | 2,1% | 129,1 |

NMI (k = 6) của A3 (K-Means trực tiếp, seed 42, TV3) để đối chiếu tác động của UMAP: B0 0,665 · B1 0,165 · B2 0,439 · B3 0,453 · B4 0,418 · B5 0,571.

## 5. Xử lý cụm nhiễu (E5, HDBSCAN, `min_cluster_size = 30`)

Tỉ lệ nhiễu 2,1% (1,5–2,8% tùy seed), chủ yếu là bài Thể thao và Giải trí nằm giữa hai cụm.

| Cách tính | ARI | NMI |
|---|---|---|
| Bỏ văn bản nhiễu | 0,432 ± 0,002 | 0,627 ± 0,001 |
| Sau `reduce_outliers()` | 0,433 ± 0,001 | 0,623 ± 0,000 |

Với tỉ lệ nhiễu thấp như vậy, gán lại nhiễu gần như không đổi kết quả.

## 6. Chủ đề tìm được (E5, seed 42)

**BERTopic k = 6** (mô hình lưu cho demo):

| Chủ đề | Số bài | Top 10 từ khóa | Nhãn thật chiếm đa số |
|---|---|---|---|
| 0 | 340 | doanh_nghiệp, việt_nam, đồng, công_nghệ, đầu_tư, giá, giảm, thị_trường, usd, tăng | Kinh tế (207) – lẫn 64 Thời sự, 57 Công nghệ |
| 1 | 325 | bác_sĩ, bệnh, điều_trị, người_bệnh, bệnh_viện, y_tế, ung_thư, thuốc, sức_khỏe, đau | Sức khỏe (281) |
| 2 | 290 | đất, đồng, thuế, phường, dân, xã, tỉnh, xe, khu_vực, tp_hcm | Thời sự (209) – lẫn 43 Kinh tế |
| 3 | 290 | diễn_viên, phim, khán_giả, văn_hóa, chương_trình, nghệ_sĩ, ca_sĩ, nghệ_thuật, diễn, nữ | Giải trí (263) |
| 4 | 278 | trận, đội, bóng, đội_tuyển, việt_nam, cầu_thủ, hlv, giải, thắng, thi_đấu | Thể thao (274) |
| 5 | 277 | nvidia, dùng, iphone, thiết_bị, huang, công_ty, hệ_thống, màn_hình, ceo, apple | Công nghệ (220) – lẫn 45 Kinh tế |

**BERTopic HDBSCAN** (`min_cluster_size = 30`):

| Chủ đề | Số bài | Top 10 từ khóa | Nhãn thật chiếm đa số |
|---|---|---|---|
| Nhiễu (-1) | 27 | — | Thể thao (14), Giải trí (13) |
| 0 | 907 | doanh_nghiệp, đồng, công_nghệ, việt_nam, đầu_tư, giá, thị_trường, công_ty, tỷ, triệu | Kinh tế 295, Công nghệ 292, Thời sự 274 (gộp 3 chuyên mục) |
| 1 | 312 | bác_sĩ, bệnh, điều_trị, người_bệnh, bệnh_viện, y_tế, ung_thư, thuốc, sức_khỏe, đau | Sức khỏe (281) |
| 2 | 286 | diễn_viên, phim, khán_giả, văn_hóa, chương_trình, nghệ_sĩ, ca_sĩ, nghệ_thuật, diễn, nữ | Giải trí (263) |
| 3 | 186 | việt_nam, đội, đội_tuyển, trận, giải, u17, bóng, thi_đấu, hlv, cầu_thủ | Thể thao (182) – bóng đá Việt Nam |
| 4 | 82 | trận, bóng, cầu_thủ, đội, ronaldo, hlv, bàn, đội_tuyển, bồ_đào_nha, sân | Thể thao (82) – bóng đá quốc tế |

## 7. Hình

| Hình | File |
|---|---|
| Bản đồ 2D (chủ đề BERTopic k = 6 / nhãn thật), màu chủ đề = màu nhãn đa số | `outputs/figures/fig1_ban_do_2d.png` |
| Top 10 từ khóa c-TF-IDF mỗi chủ đề (k = 6) | `outputs/figures/fig2_tu_khoa.png` |
| Heatmap chủ đề × nhãn thật (k = 6 và HDBSCAN, cạnh nhau) | `outputs/figures/fig3_heatmap.png` |
| Biểu đồ cột nhóm Bảng 1 (thanh lỗi = độ lệch chuẩn 3 seed) | `outputs/figures/fig4_bang1.png` |
| NMI theo thời gian embedding (Bảng 2), đường gạch = LSA | `outputs/figures/fig5_nmi_thoi_gian.png` |

## 8. Thí nghiệm bổ sung (nếu thực hiện)

| Thí nghiệm | Kết quả |
|---|---|
| P1. Độ dài đầu vào | Chưa thực hiện (cần tính lại embedding cho từng phiên bản đầu vào) |
| P2. Tập con 20 bài/chủ đề | Chưa thực hiện |

## 9. Nhận xét

- **BERTopic (k = 6) tốt nhất về mức khớp nhãn, nhưng không áp đảo.** So với LSA + K-Means: ARI cao hơn rõ (0,623 so với 0,572), Purity cao hơn (0,809 so với 0,786), NMI ngang nhau (0,647 so với 0,651, trong phạm vi độ lệch chuẩn). Đổi lại chậm hơn ~66 lần (133 s so với 2 s) và C_v thấp hơn (0,711 so với 0,792). Đúng như rủi ro dự kiến ở KE_HOACH §13: tin tức có từ khóa đặc trưng rõ nên phương pháp tần suất đã rất mạnh.
- **BERTopic ổn định hơn hẳn giữa các seed**: độ lệch chuẩn NMI 0,004, so với 0,015 (LSA), 0,020 (A3) và 0,022 (LDA).
- **Phân tích đóng góp (KE_HOACH §6.1):**
  - A2 → A3 (đổi TF-IDF sang embedding E5, giữ K-Means): NMI giảm 0,651 → 0,584. Riêng biểu diễn BERT (chỉ tiêu đề + sapo) chưa thắng TF-IDF toàn văn.
  - A3 → A4 k = 6 (thêm UMAP 5 chiều trước K-Means): NMI 0,584 → 0,647, ARI 0,531 → 0,623. **UMAP là bước tạo ra phần lớn cải thiện.** UMAP giúp mọi embedding BERT ở k = 6 (B1 0,165 → 0,376; B3 0,453 → 0,591; B4 0,418 → 0,583), nhưng làm B0 giảm nhẹ (0,665 → 0,626).
  - A4 k = 6 → A4 HDBSCAN: tự xác định số chủ đề, nhưng ở cấu hình chọn theo C_v thì gộp Thời sự + Kinh tế + Công nghệ thành một cụm 907 bài, nên ARI giảm còn 0,432 dù NMI gần giữ nguyên (0,627).
- **Hướng B lặp lại xếp hạng của TV3 khi đổi sang BERTopic:** E5 > MiniLM ≈ Vietnamese-SBERT > mBERT mean > mBERT `[CLS]`. Lợi thế của mô hình đã huấn luyện cho câu (B3–B5) so với BERT nguyên bản (B1, B2) vẫn rõ.
- **`[CLS]` của mBERT sụp đổ với HDBSCAN:** chỉ ra 2 cụm, NMI 0,01. Vector `[CLS]` khi chưa fine-tune gần như giống nhau giữa mọi bài, nên phân cụm theo mật độ không tách được; K-Means buộc phải chia 6 cụm nên vẫn đạt 0,376. Đây là minh chứng trực quan cho lý thuyết T4.
- **HDBSCAN tìm chủ đề con có nghĩa:** Thể thao tách thành bóng đá Việt Nam (U17, đội tuyển) và bóng đá quốc tế (Ronaldo, Bồ Đào Nha), đúng hiện tượng dự đoán ở KE_HOACH §7.4. Với `min_cluster_size = 15`, E5 cho 18 chủ đề nhỏ hơn nữa.
- **Nhầm lẫn còn lại giữa Thời sự – Kinh tế – Công nghệ**, như ở LDA/LSA nhưng nhẹ hơn: BERTopic k = 6 tách được một chủ đề Thời sự riêng (đất, thuế, phường, xã: tin địa phương, hành chính; 209/290 bài), trong khi LSA dồn 285 bài Thời sự vào cụm 589 bài. Chủ đề "Nvidia" vẫn xuất hiện, nhưng gộp với thiết bị (iPhone, Apple) thành chủ đề Công nghệ (220/277 bài).
- **Chi phí:** khi đã có embedding, BERTopic chỉ mất 4–9 s; gần như toàn bộ chi phí là bước embedding. MiniLM (37 s) là lựa chọn rẻ với NMI 0,591; E5 (129 s) tốt nhất.
- **Hạn chế:** BERT chỉ nhận tiêu đề + sapo, còn TF-IDF dùng toàn văn; `min_cluster_size` chọn theo C_v ưu tiên ít cụm lớn; kết quả trên một bộ dữ liệu thu thập trong khoảng thời gian ngắn.
- **Tái lập kết quả:** chế độ k = 6 cho cùng kết quả trên mọi máy. Chế độ HDBSCAN phụ thuộc hệ điều hành: macOS cho đúng số trong `results.csv` (5 chủ đề), Linux cho 10–12 chủ đề dù cùng phiên bản thư viện. Số chính thức là số trong `results.csv`.

## Dàn ý slide

> Mỗi slide: tiêu đề, 2–4 ý chính, bảng/hình sử dụng, điểm cần nhấn mạnh khi thuyết trình.

### Slide 1 – BERTopic so với phương pháp truyền thống (Bảng 1)

- **Ý chính:**
  - Cùng k = 6, 3 seed: BERTopic đạt ARI 0,62 và Purity 0,81, cao nhất; NMI 0,65 ngang LSA.
  - LSA + K-Means vẫn rất mạnh: NMI 0,65, C_v cao nhất, chỉ 2 giây; BERTopic mất ~133 giây (gần hết là thời gian embedding).
  - BERTopic ổn định nhất giữa các seed (độ lệch NMI 0,004).
- **Bảng / hình:** `outputs/figures/fig4_bang1.png` (hoặc bảng mục 3 rút gọn ARI, NMI, C_v, thời gian)
- **Điểm nhấn:** BERT không tự động thắng; với tin tức có từ khóa rõ, TF-IDF đã rất tốt. Lợi thế của BERTopic là ARI, độ ổn định và khả năng tìm chủ đề con.

### Slide 2 – Bước nào tạo ra cải thiện? (phân tích đóng góp)

- **Ý chính:**
  - LSA → BERT + K-Means: chỉ đổi biểu diễn, NMI giảm (0,65 → 0,58).
  - BERT + K-Means → BERTopic k = 6: thêm UMAP, NMI tăng (0,58 → 0,65), ARI tăng (0,53 → 0,62).
  - BERTopic k = 6 → HDBSCAN: tự xác định số chủ đề nhưng gộp Thời sự + Kinh tế + Công nghệ (ARI giảm còn 0,43).
- **Bảng / hình:** sơ đồ mũi tên A2 → A3 → A4 kèm NMI, ARI từ bảng mục 3
- **Điểm nhấn:** UMAP là mảnh ghép quan trọng của BERTopic, không chỉ là embedding BERT.

### Slide 3 – Hướng B: cách lấy biểu diễn trong BERTopic

- **Ý chính:**
  - Xếp hạng giữ nguyên như K-Means: E5 (0,65) > MiniLM, Vi-SBERT (~0,59) > mBERT mean (0,51) > mBERT `[CLS]` (0,38).
  - `[CLS]` chưa fine-tune: HDBSCAN chỉ tìm được 2 cụm (NMI 0,01) vì mọi vector gần giống nhau.
  - MiniLM rẻ nhất (37 s) mà gần bằng các mô hình lớn hơn.
- **Bảng / hình:** `outputs/figures/fig5_nmi_thoi_gian.png`
- **Điểm nhấn:** khớp lý thuyết T4: mean pooling tốt hơn `[CLS]`, Sentence-BERT tốt hơn BERT nguyên bản.

### Slide 4 – Chủ đề BERTopic tìm được

- **Ý chính:**
  - 6 chủ đề khớp 6 chuyên mục: Thể thao 274/278, Sức khỏe 281/325, Giải trí 263/290.
  - Thời sự tách được thành chủ đề "địa phương, hành chính" (đất, thuế, phường, xã), tốt hơn LSA.
  - HDBSCAN tách Thể thao thành bóng đá Việt Nam và bóng đá quốc tế (Ronaldo).
- **Bảng / hình:** `outputs/figures/fig1_ban_do_2d.png`, `outputs/figures/fig3_heatmap.png` (có thể thêm `fig2_tu_khoa.png`)
- **Điểm nhấn:** nhầm lẫn còn lại nằm ở các chuyên mục chồng lấn tự nhiên (Kinh tế – Công nghệ – Thời sự); chuyển sang demo.
