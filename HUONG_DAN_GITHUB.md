# Hướng dẫn dùng GitHub cho thành viên

Tài liệu dành cho thành viên **chưa từng dùng GitHub**. Đọc hết một lần trước khi bắt đầu, sau đó dùng [Quy trình mỗi lần làm việc](#3-quy-trình-mỗi-lần-làm-việc) và [Checklist](#6-checklist-trước-khi-tạo-pull-request) là đủ.

- **Repo:** https://github.com/anhdung1106/NLP-BERT-Topic-Modeling
- **Công cụ:** VS Code (bấm nút trên giao diện). Mỗi bước có kèm lệnh Terminal tương đương để tham khảo.
- **Nguyên tắc quan trọng nhất:** chỉ làm trên **branch của mình**, không bao giờ push thẳng vào `main`.

---

## 1. Hiểu nhanh

| Khái niệm | Hiểu đơn giản |
|---|---|
| **Repo** | Thư mục dự án dùng chung, lưu trên GitHub |
| **`main`** | Bản chính thức. Ai tải về cũng phải chạy được. **Chỉ TV5 gộp code vào đây** |
| **Branch** | Bản nháp riêng của mỗi người, tách ra từ `main`. Sửa thoải mái mà không ảnh hưởng người khác |
| **Commit** | Một lần "lưu có ghi chú" trên máy mình |
| **Push** | Đẩy các commit từ máy mình lên GitHub |
| **Pull** | Kéo thay đổi mới nhất từ GitHub về máy mình |
| **Pull Request (PR)** | Đề nghị TV5 xem và gộp branch của mình vào `main` |

```
main ──────────────●───────────────●────────── (bản chính thức)
                    \             ↑ TV5 merge PR
feat/data            ●───●───●───┘  (commit của bạn, push lên, tạo PR)
```

---

## 2. Chuẩn bị (làm một lần)

### 2.1. Tài khoản & quyền truy cập

1. Tạo tài khoản tại https://github.com (nếu chưa có).
2. Gửi **username GitHub** cho TV5.
3. Mở email (hoặc https://github.com/notifications), bấm **Accept invitation** để được quyền push. Chưa chấp nhận lời mời thì push sẽ bị lỗi `403`.

### 2.2. Cài đặt

| Phần mềm | Windows | macOS |
|---|---|---|
| **Git** | Tải tại https://git-scm.com/download/win, cài với tùy chọn mặc định | Mở Terminal, gõ `git --version`. Nếu máy hỏi cài "command line developer tools", bấm **Install** |
| **VS Code** | https://code.visualstudio.com | https://code.visualstudio.com |
| **Python 3.11** | https://www.python.org/downloads (tick **Add Python to PATH**) | https://www.python.org/downloads |

### 2.3. Khai báo tên và email cho Git

Mở Terminal trong VS Code (**Terminal → New Terminal**), gõ 2 lệnh (thay bằng thông tin của bạn, email phải **trùng email tài khoản GitHub**):

```bash
git config --global user.name "Ten Cua Ban"
git config --global user.email "email_github_cua_ban@gmail.com"
```

### 2.4. Clone repo về máy

**VS Code:**
1. Nhấn `Ctrl+Shift+P` (Windows) hoặc `Cmd+Shift+P` (macOS), gõ **Git: Clone**, Enter.
2. Dán link `https://github.com/anhdung1106/NLP-BERT-Topic-Modeling.git`, Enter.
3. Chọn thư mục lưu trên máy → bấm **Open** khi VS Code hỏi.
4. Nếu VS Code yêu cầu đăng nhập GitHub, bấm **Allow** và đăng nhập trên trình duyệt.

> Terminal: `git clone https://github.com/anhdung1106/NLP-BERT-Topic-Modeling.git`

### 2.5. Tạo branch của mình

Mỗi người **một branch cố định** trong suốt dự án:

| Thành viên | Branch |
|---|---|
| TV1 | `feat/data` |
| TV2 | `feat/baselines` |
| TV3 | `feat/embedding` |
| TV4 | `feat/bertopic` |
| TV5 | `feat/app` |
| TV6 – TV10 | `docs/tv6`, `docs/tv7`, `docs/tv8`, `docs/tv9`, `docs/tv10` |

**VS Code:**
1. Bấm vào chữ **`main`** ở **góc dưới bên trái** cửa sổ.
2. Chọn **+ Create new branch...**, gõ tên branch (ví dụ `feat/data`), Enter.
3. Góc dưới bên trái giờ hiện tên branch của bạn. Vào tab **Source Control** (biểu tượng nhánh cây, thanh bên trái), bấm **Publish Branch** để đưa branch lên GitHub.

> Terminal: `git switch -c feat/data` rồi `git push -u origin feat/data`

### 2.6. Cài thư viện (chỉ nhóm code)

Làm theo mục **Cài đặt** trong [README.md](README.md).

---

## 3. Quy trình mỗi lần làm việc

```
① Kiểm tra branch → ② Cập nhật từ main → ③ Làm việc → ④ Commit → ⑤ Push → ⑥ Tạo PR (khi xong một phần)
```

### ① Kiểm tra đang ở branch của mình

Nhìn **góc dưới bên trái** VS Code. Phải là branch của bạn (ví dụ `feat/data`), **không phải `main`**. Nếu sai, bấm vào đó và chọn đúng branch.

> Terminal: `git branch` (dòng có dấu `*` là branch hiện tại); đổi branch: `git switch feat/data`

### ② Cập nhật phần người khác đã làm

Lấy những gì đã được gộp vào `main` về branch của mình, để không làm trên bản cũ.

**VS Code:** tab **Source Control** → bấm **`···`** (More Actions) → **Pull, Push** → **Pull from...** → chọn **`origin/main`**.

> Terminal: `git pull origin main`

Nếu xuất hiện conflict, xem [mục 5](#5-xử-lý-lỗi-thường-gặp).

### ③ Làm việc

- Chỉ sửa **file thuộc phần mình** theo [PHAN_CONG.md](PHAN_CONG.md).
- Cần sửa file của người khác (ví dụ đổi cột trong `articles.csv`): **báo người đó và TV5 trước**.

### ④ Commit

**VS Code:**
1. Vào tab **Source Control**. Danh sách **Changes** là các file bạn đã sửa.
2. Bấm vào từng file để xem lại thay đổi. Kiểm tra không có file lạ (xem [mục 4](#4-quy-tắc)).
3. Rê chuột vào file cần lưu, bấm dấu **`+`** (stage). File chuyển sang mục **Staged Changes**.
4. Gõ **lời mô tả** vào ô **Message** phía trên (cách viết ở [mục 4](#4-quy-tắc)).
5. Bấm **Commit**.

> Terminal: `git add <ten_file>` rồi `git commit -m "Thêm crawler cho VnExpress"`

Nên commit **nhỏ và thường xuyên**: mỗi khi xong một việc nhỏ, chạy được, thì commit.

### ⑤ Push

**VS Code:** bấm **Sync Changes** (hoặc biểu tượng mũi tên ↑ ở góc dưới bên trái).

> Terminal: `git push`

Nên push **ít nhất một lần mỗi ngày**, kể cả khi chưa xong, để TV5 theo dõi tiến độ và không mất code nếu máy hỏng.

### ⑥ Tạo Pull Request (khi xong một phần việc)

1. Mở trang repo trên GitHub. Thường sẽ có thanh màu vàng **"feat/data had recent pushes"** → bấm **Compare & pull request**.
   Nếu không thấy: tab **Pull requests** → **New pull request** → chọn **base: `main`** ← **compare: branch của bạn**.
2. Kiểm tra mũi tên là **`base: main` ← `compare: <branch của bạn>`**.
3. **Title:** mô tả ngắn, ví dụ `[TV1] Crawler + tiền xử lý dữ liệu`.
4. **Description:** điền theo mẫu:

   ```
   ## Đã làm
   - 

   ## File thay đổi chính
   - 

   ## Cách kiểm tra
   - Chạy: 
   - Kết quả mong đợi: 

   ## Ghi chú / cần TV5 lưu ý
   - 
   ```

5. Bấm **Create pull request**, rồi **nhắn TV5** link PR.
6. **Không tự bấm Merge.** Nếu TV5 yêu cầu sửa: sửa trên branch của mình → commit → push. PR tự cập nhật, không cần tạo PR mới.
7. Sau khi PR được merge: **tiếp tục làm trên branch cũ**, bắt đầu lại từ bước ①.

---

## 4. Quy tắc

### Lời mô tả commit

- Tiếng Việt, ngắn gọn (dưới ~70 ký tự), **bắt đầu bằng động từ**, nói rõ đã làm gì.

| ✅ Nên | ❌ Không nên |
|---|---|
| `Thêm crawler cho VnExpress và Tuổi Trẻ` | `update` |
| `Sửa lỗi tách từ với câu có số` | `sửa` |
| `Tính embedding B3 MiniLM, lưu b3_minilm.npy` | `abc`, `final`, `final2` |

### Không được commit

| Không commit | Lý do |
|---|---|
| `.venv/`, `__pycache__/`, `.ipynb_checkpoints/` | File môi trường của riêng máy bạn (đã có `.gitignore` chặn sẵn) |
| `data/raw/` | Dữ liệu thô, nặng (đã chặn sẵn) |
| File trên **50 MB** | GitHub từ chối file trên 100 MB; file lớn làm repo chậm cho cả nhóm |
| Mật khẩu, API key | Bảo mật |
| Video demo | Gửi qua Google Drive |

Nếu thấy các file trên trong danh sách **Changes**: **không bấm `+`** cho file đó và hỏi TV5.

### Notebook (`.ipynb`)

- Trước khi commit: **Restart → Run All** để output đúng thứ tự và chạy được từ đầu đến cuối.
- Commit **kèm output** để cả nhóm xem kết quả trên GitHub.

### Tuyệt đối không

- Push hoặc commit khi đang ở branch **`main`**.
- Tự bấm **Merge** Pull Request.
- Dùng `git push --force` hoặc các lệnh có chữ `--force`, `reset --hard` khi chưa hỏi TV5.
- Đổi tên cột, tên file trong [QUY_UOC_CODE.md](QUY_UOC_CODE.md) mà không báo cả nhóm.

---

## 5. Xử lý lỗi thường gặp

| Tình huống | Cách xử lý |
|---|---|
| Push báo **`rejected`** / **`fetch first`** | Trên GitHub có thay đổi mà máy bạn chưa có. Bấm **Pull** (hoặc `git pull`), rồi push lại |
| Push báo **`403`** / **`Permission denied`** | Chưa chấp nhận lời mời vào repo (mục 2.1), hoặc đăng nhập sai tài khoản. Kiểm tra biểu tượng **Accounts** (góc dưới thanh bên trái VS Code) |
| VS Code báo **`Please tell me who you are`** | Chưa khai báo tên và email (mục 2.3) |
| **Conflict** khi pull | Xem bên dưới |
| Đã sửa file nhưng **quên chuyển sang branch của mình** (chưa commit) | Bấm tên branch ở góc dưới bên trái → chọn **branch của mình**. Thay đổi chưa commit sẽ đi theo. Nếu VS Code không cho chuyển, hỏi TV5 |
| **Đã commit vào `main`** trên máy | **Không push.** Nhắn TV5 để xử lý |
| Lỡ commit file lớn hoặc file không được commit | **Không push.** Nhắn TV5 |
| Không biết mình đang ở đâu, nhiều lỗi lạ | Dừng lại, chụp màn hình tab **Source Control** và thông báo lỗi, gửi TV5. Đừng thử lệnh lạ trên mạng |

### Conflict là gì, xử lý thế nào

Conflict xảy ra khi **bạn và người khác cùng sửa một chỗ** trong cùng một file. Git không biết giữ bản nào nên hỏi bạn.

1. VS Code đánh dấu file bị conflict bằng chữ **`!`** trong tab **Source Control**. Mở file đó.
2. Mỗi chỗ conflict có hai phần:
   - **Current Change:** bản của bạn
   - **Incoming Change:** bản từ `main`
3. Bấm một trong các lựa chọn hiện phía trên đoạn đó:
   - **Accept Current Change:** giữ bản của bạn
   - **Accept Incoming Change:** giữ bản từ `main`
   - **Accept Both Changes:** giữ cả hai
4. Xử lý hết các chỗ, **lưu file**, bấm **`+`** cho file đó, rồi **Commit** như bình thường.

**Nếu conflict nằm trong file không phải của bạn, hoặc bạn không chắc chọn bản nào: dừng lại và hỏi TV5.**

---

## 6. Checklist trước khi tạo Pull Request

- [ ] Đang ở **branch của mình**, không phải `main`
- [ ] Đã **pull từ `main`** (bước ②) và không còn conflict
- [ ] Code / notebook **chạy được từ đầu đến cuối** trên máy mình
- [ ] Chỉ có **file thuộc phần mình** trong danh sách thay đổi
- [ ] Không có file nặng, `.venv/`, `data/raw/`, video
- [ ] File đầu ra **đúng tên và định dạng** theo [QUY_UOC_CODE.md](QUY_UOC_CODE.md)
- [ ] Đã cập nhật file của mình trong `ket_qua/` (nhóm code) hoặc `slides/` (nhóm slide, thuyết trình)
- [ ] Lời mô tả commit rõ ràng
- [ ] Đã **push** lên GitHub

---

## 7. Cách nhanh cho người chỉ sửa file `.md` (nhóm slide, thuyết trình)

Nếu chỉ cần sửa file `.md` trong `slides/`, có thể làm **trực tiếp trên trang GitHub**, không cần cài Git hay clone:

1. Mở file cần sửa trên GitHub (ví dụ `slides/script_thuyet_trinh.md`).
2. Bấm biểu tượng **bút chì** (Edit this file) ở góc phải.
3. Sửa nội dung. Tab **Preview** để xem trước định dạng.
4. Bấm **Commit changes...**:
   - Ô **Commit message:** mô tả ngắn, ví dụ `Thêm script phần kết quả`
   - Chọn **Create a new branch for this commit and start a pull request**
   - Đặt tên branch theo mục 2.5 (ví dụ `docs/tv9`)
5. Bấm **Propose changes** → **Create pull request**, rồi nhắn TV5.
