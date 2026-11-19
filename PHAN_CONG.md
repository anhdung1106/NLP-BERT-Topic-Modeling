# Phân công công việc

> Điền tên thành viên vào cột **Họ tên**. Chi tiết phương pháp xem [KE_HOACH.md](KE_HOACH.md).

## Tổng quan

| TV | Họ tên | Vai trò | Mảng công việc | File `ket_qua/` / `slides/` phụ trách |
|---|---|---|---|---|
| TV1 | _điền tên_ | Lập trình | Dữ liệu: thu thập, tiền xử lý, thống kê mô tả | `ket_qua/01_du_lieu.md` |
| TV2 | _điền tên_ | Lập trình | Mô hình truyền thống (A1, A2), bộ chỉ số đánh giá, phân tích lỗi | `ket_qua/02_mo_hinh_truyen_thong.md` |
| TV3 | _điền tên_ | Lập trình | Biểu diễn BERT (B0–B5), A3 | `ket_qua/03_bieu_dien_bert.md` |
| TV4 | _điền tên_ | Lập trình | BERTopic, Hướng B, A4, bảng và hình kết quả | `ket_qua/04_bertopic_so_sanh.md` |
| TV5 | _điền tên_ | Lập trình | Repo, review và merge, ứng dụng demo, video dự phòng | `ket_qua/05_demo.md`, `slides/kich_ban_demo.md` |
| TV6 | _điền tên_ | Slide | Template, slide lý thuyết | `slides/bo_cuc_slide.md` (phần lý thuyết) |
| TV7 | _điền tên_ | Slide | Slide dữ liệu và phương pháp | `slides/bo_cuc_slide.md` (phần dữ liệu, phương pháp) |
| TV8 | _điền tên_ | Slide | Slide kết quả, demo, kết luận | `slides/bo_cuc_slide.md` (phần kết quả, kết luận) |
| TV9 | _điền tên_ | Thuyết trình | Mở đầu, lý thuyết, dữ liệu & phương pháp (~6 phút) | `slides/script_thuyet_trinh.md` (phần 1) |
| TV10 | _điền tên_ | Thuyết trình | Kết quả, demo, kết luận (~9 phút) | `slides/script_thuyet_trinh.md` (phần 2) |

Dàn ý lý thuyết: `slides/dan_y_ly_thuyet.md`, hoàn tất **cuối ngày 2**.

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
| Bộ chỉ số dùng chung: ARI, NMI, Purity, C_v, diversity (ngày 2) | `src/metrics.py` (hàm `evaluate`, xem [quy ước](#quy-ước-file-trao-đổi)) |
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
| Cấu hình chung (ngày 1) | `src/config.py` (đường dẫn, seed, danh sách mô hình) |
| Review và merge Pull Request | Nhánh `main` luôn chạy được |
| Ứng dụng demo 4 tab | `app/streamlit_app.py` |
| Chốt phiên bản thư viện, kiểm thử trên macOS và Windows | `requirements.txt`, mục cài đặt trong `README.md` |
| Kịch bản demo, video dự phòng | `slides/kich_ban_demo.md`, video (không commit, gửi qua Drive) |

---

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

Chốt trong **ngày 1**. Không đổi tên cột hoặc định dạng khi chưa báo cả nhóm.

| File | Định dạng | Người tạo | Người dùng |
|---|---|---|---|
| `data/processed/articles.csv` | `id, title, sapo, content, url, source, date, true_label, text_tok, text_bert, text_bert_seg` (UTF-8) | TV1 | Tất cả |
| `embeddings/<model>.npy` | Ma trận (số văn bản × số chiều), **thứ tự dòng trùng với `articles.csv`** | TV3 | TV4, TV5 |
| `outputs/tables/results.csv` | `method, embedding, mode, seed, n_topics, pct_noise, ari, nmi, purity, cv, diversity, time_s` | TV2, TV3, TV4 | TV5, nhóm slide |
| `outputs/tables/doc_topics.csv` | `id, <model>_topic, x, y` (x, y: tọa độ 2D để vẽ bản đồ) | TV4 | TV5 |
| `src/metrics.py` → `evaluate(labels_pred, labels_true, topic_words, texts) -> dict` | Trả về `ari, nmi, purity, cv, diversity` | TV2 | TV3, TV4 |

## Quy ước Git

| Thành viên | Branch |
|---|---|
| TV1 | `feat/data` |
| TV2 | `feat/baselines` |
| TV3 | `feat/embedding` |
| TV4 | `feat/bertopic` |
| TV5 | `feat/app` |
| TV6–TV10 | Sửa file `.md` trong `slides/` qua branch `docs/slides` |

- Merge vào `main` qua Pull Request, TV5 review.
- Notebook commit **kèm output** để xem kết quả trên GitHub.
- Không commit `data/raw/`, video.
