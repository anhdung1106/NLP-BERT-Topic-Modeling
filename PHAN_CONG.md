# Phân công công việc

> Điền tên thành viên vào cột **Họ tên**. Chi tiết phương pháp xem [KE_HOACH.md](KE_HOACH.md). Tên hàm, tên file, tên cột của từng phần xem [QUY_UOC_CODE.md](QUY_UOC_CODE.md).

## Tổng quan

| TV | Họ tên | Vai trò | Mảng công việc | File `ket_qua/` / `slides/` phụ trách | Trạng thái |
|---|---|---|---|---|---|
| TV1 | _điền tên_ | Lập trình | Dữ liệu: thu thập, tiền xử lý, thống kê mô tả | `ket_qua/01_du_lieu.md` | Xong |
| TV2 | _Thanh Hằng_ | Lập trình | Mô hình truyền thống (A1, A2), bộ chỉ số đánh giá, phân tích lỗi | `ket_qua/02_mo_hinh_truyen_thong.md` | Đang làm (phân tích lỗi) |
| TV3 | _điền tên_ | Lập trình | Biểu diễn BERT (B0–B5), A3 | `ket_qua/03_bieu_dien_bert.md` | Xong |
| TV4 | _điền tên_ | Lập trình | BERTopic, Hướng B, A4, bảng và hình kết quả | `ket_qua/04_bertopic_so_sanh.md` | Xong |
| TV5 | _điền tên_ | Lập trình | Quản lý repo, duyệt và merge PR, ứng dụng demo, video dự phòng | `ket_qua/05_demo.md`, `slides/kich_ban_demo.md` | Xong |
| TV6 | _điền tên_ | Slide | Template, slide lý thuyết | `slides/bo_cuc_slide.md` (phần lý thuyết) | Chưa bắt đầu |
| TV7 | _điền tên_ | Slide | Slide dữ liệu và phương pháp | `slides/bo_cuc_slide.md` (phần dữ liệu, phương pháp) | Chưa bắt đầu |
| TV8 | _điền tên_ | Slide | Slide kết quả, demo, kết luận | `slides/bo_cuc_slide.md` (phần kết quả, kết luận) | Chưa bắt đầu |
| TV9 | _điền tên_ | Thuyết trình | Mở đầu, lý thuyết, dữ liệu & phương pháp (~6 phút) | `slides/script_thuyet_trinh.md` (phần 1) | Chưa bắt đầu |
| TV10 | _điền tên_ | Thuyết trình | Kết quả, demo, kết luận (~9 phút) | `slides/script_thuyet_trinh.md` (phần 2) | Chưa bắt đầu |

Dàn ý lý thuyết: `slides/dan_y_ly_thuyet.md`, hoàn tất **cuối ngày 2**.

**Cập nhật trạng thái:** mỗi người tự sửa cột *Trạng thái* của mình (`Chưa bắt đầu` → `Đang làm` → `Đã tạo PR` → `Xong`) mỗi khi đổi trạng thái. Gặp vướng quá nửa ngày thì ghi `Vướng: <lý do>` và nhắn TV5.

---

## Nhóm lập trình

### TV1 – Dữ liệu · hạn: **cuối ngày 2** (đường găng)

| Công việc | Đầu ra (file) |
|---|---|
| Chạy thử thu thập 2 trang mỗi báo, kiểm tra phân trang (ngày 1) | Danh sách báo dùng được, ghi trong `ket_qua/01_du_lieu.md` |
| Dữ liệu mẫu cho các thành viên khác (ngày 1) | `data/processed/articles_sample.csv` (~20 bài/chủ đề, đúng định dạng) |
| Script thu thập 6 chủ đề × 4 báo | `src/crawl.py` → `data/raw/<chu_de>.jsonl` |
| Tiền xử lý, gộp, loại trùng, cân bằng | `src/preprocess.py`, `data/stopwords/vietnamese_stopwords.txt` → `data/processed/articles.csv` |
| Thống kê mô tả (số bài/chủ đề, độ dài, nguồn) | `notebooks/01_du_lieu.ipynb`, hình trong `outputs/figures/` |
| Kết quả + dàn ý slide | `ket_qua/01_du_lieu.md` |

### TV2 – Mô hình truyền thống & đánh giá · hạn: **cuối ngày 4**

| Công việc | Đầu ra (file) |
|---|---|
| Bộ chỉ số dùng chung: ARI, NMI, Purity, C_v, diversity (ngày 2) | `src/metrics.py` (hàm `evaluate`, xem [QUY_UOC_CODE.md](QUY_UOC_CODE.md) §4.3) |
| LDA (A1), LSA + K-Means (A2), quét số chiều SVD | `src/baselines.py`, `notebooks/02_mo_hinh_truyen_thong.ipynb` |
| Lưu mô hình LDA cho demo | `models/lda/` |
| Ghi kết quả A1, A2 | Các dòng tương ứng trong `outputs/tables/results.csv` |
| Phân tích lỗi LDA và BERTopic (ngày 4) | Mục phân tích lỗi trong `ket_qua/02_mo_hinh_truyen_thong.md` |

### TV3 – Biểu diễn BERT · hạn: **cuối ngày 3**

| Công việc | Đầu ra (file) |
|---|---|
| Trích xuất embedding B0–B5, đo thời gian | `src/embed.py`, `notebooks/03_bieu_dien_bert.ipynb` |
| Lưu và commit embedding | `embeddings/b0_tfidf_svd.npy`, `b1_mbert_cls.npy`, `b2_mbert_mean.npy`, `b3_minilm.npy`, `b4_vi_sbert.npy`, `b5_e5.npy` |
| Bảng thời gian embedding | `outputs/tables/embedding_time.csv` |
| BERT + K-Means (A3) | Các dòng tương ứng trong `outputs/tables/results.csv` |

### TV4 – BERTopic & tổng hợp kết quả · hạn: **cuối ngày 4** (hình: cuối ngày 5)

| Công việc | Đầu ra (file) |
|---|---|
| Pipeline BERTopic, 2 chế độ (HDBSCAN, K-Means k = 6) | `src/bertopic_runner.py` |
| Hướng B, quét `min_cluster_size`, A4, 3 seed | `notebooks/04_bertopic_so_sanh.ipynb` |
| Bảng kết quả, chủ đề từng văn bản | `outputs/tables/results.csv`, `outputs/tables/doc_topics.csv` |
| Lưu mô hình BERTopic cho demo | `models/bertopic_best/` |
| 5 hình theo KE_HOACH §7.3 | `outputs/figures/` |

### TV5 – Repo, tích hợp & demo · hạn: **cuối ngày 5**

| Công việc | Đầu ra (file) |
|---|---|
| Cấu hình chung | `src/config.py` (đã xong) |
| Duyệt và merge Pull Request, kiểm tra `main` chạy được sau mỗi lần merge | Nhánh `main` luôn chạy được |
| Theo dõi cột *Trạng thái*, nhắc hạn | Bảng tổng quan ở đầu file này |
| Ứng dụng demo 4 tab | `app/streamlit_app.py` |
| Chốt phiên bản thư viện, kiểm thử trên macOS và Windows | `requirements.txt`, mục cài đặt trong `README.md` |
| Kịch bản demo, video dự phòng | `slides/kich_ban_demo.md`, video (không commit, gửi qua Drive) |

---

## Để không phải chờ nhau

| Người | Chưa có gì thì làm gì |
|---|---|
| TV2, TV3, TV4, TV5 | Viết và chạy thử code trên `load_articles(sample=True)` trong lúc TV1 crawl |
| TV3, TV4 | Chưa có `metrics.py` của TV2: viết hàm tạm **cùng tên, cùng đầu vào, đầu ra** như [QUY_UOC_CODE.md](QUY_UOC_CODE.md) §4.3, có bản thật thì xóa bản tạm |
| TV4 | Chưa có embedding của TV3: dùng tạm B0 (`embed_tfidf_svd`) hoặc ma trận ngẫu nhiên đúng số dòng để chạy thử BERTopic |
| TV5 | Chưa có kết quả: tự tạo `doc_topics.csv`, `results.csv` giả **đúng tên cột** để dựng app |
| TV6–TV8 | Chưa có kết quả: dựng khung slide theo [KE_HOACH.md](KE_HOACH.md), để trống chỗ bảng và hình |

## Nhóm slide

| TV | Công việc | Nguồn nội dung | Hạn |
|---|---|---|---|
| TV6 | Template (màu, font, bố cục); slide lý thuyết T1–T5 | `slides/dan_y_ly_thuyet.md` | Template: ngày 1 · lý thuyết: ngày 3 |
| TV7 | Slide dữ liệu, phương pháp (Hướng A, B, bộ chỉ số) | `ket_qua/01`, `02`, `03` | Ngày 5 |
| TV8 | Slide kết quả, demo, kết luận; định dạng bảng và hình cho máy chiếu | `ket_qua/04`, `05`, `outputs/` | Ngày 5 |

Cả ba cùng cập nhật `slides/bo_cuc_slide.md`. Bản slide hoàn chỉnh: **cuối ngày 6**.

## Nhóm thuyết trình

| TV | Phần trình bày | Chuẩn bị | Hạn |
|---|---|---|---|
| TV9 | Mở đầu, lý thuyết, dữ liệu & phương pháp (~6 phút) | Nắm `slides/dan_y_ly_thuyet.md`, `ket_qua/01–03`; cài repo, chạy thử demo | Script: ngày 6 |
| TV10 | Kết quả, demo, kết luận (~9 phút) | Nắm `ket_qua/04–05`; thực hành demo theo `slides/kich_ban_demo.md` tối thiểu 3 lần | Script: ngày 6 |

Hai thành viên cùng soạn **câu hỏi dự kiến** trong `slides/script_thuyet_trinh.md`, phối hợp với thành viên lập trình tương ứng để chuẩn bị câu trả lời.

---

## Quy ước file trao đổi

Tên hàm, hằng số, cột dữ liệu, tên file đầu ra của từng phần được quy định **cố định** trong [QUY_UOC_CODE.md](QUY_UOC_CODE.md). Mọi thành viên lập trình đọc file này trước khi viết code.

## Quy ước Git

| Thành viên | Branch |
|---|---|
| TV1 | `feat/data` |
| TV2 | `feat/baselines` |
| TV3 | `feat/embedding` |
| TV4 | `feat/bertopic` |
| TV5 | `feat/app` |
| TV6–TV10 | `docs/tv6` … `docs/tv10` (sửa file `.md` trong `slides/`) |

- Merge vào `main` qua Pull Request. Cách làm từng bước: [HUONG_DAN_GITHUB.md](HUONG_DAN_GITHUB.md).
- Notebook commit **kèm output** để xem kết quả trên GitHub.
- Không commit `data/raw/`, video.
