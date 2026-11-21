"""Tiền xử lý: đọc data/raw/*.jsonl -> data/processed/articles.csv (3 phiên bản văn bản).

Chạy từ thư mục gốc repo:
    python -m src.preprocess            # tạo articles.csv (dữ liệu chính thức)
    python -m src.preprocess --sample   # tạo articles_sample.csv (~20 bài/chủ đề) cho cả nhóm dùng thử
"""
import argparse
import html
import json
import re
import unicodedata

import pandas as pd
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (ARTICLE_COLUMNS, ARTICLES_CSV, ARTICLES_PER_TOPIC, DATA_RAW,
                        MIN_ARTICLES_PER_TOPIC, MIN_WORDS, NEAR_DUP_THRESHOLD, SAMPLE_CSV, SAMPLE_PER_TOPIC, SEED, STOPWORDS_TXT,
                        TOPICS, VECTORIZER_PARAMS)

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://|www\.)\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")  # email tòa soạn / tác giả ở cuối bài
_TPHCM_RE = re.compile(r"\bTP\s*\.?\s*HCM\b", re.I)  # TP.HCM, TP. HCM, TPHCM, TP HCM
_TY_RE = re.compile(r"\b[Tt]ỉ\b(?!\s+(?:mỉ|tê)\b)")  # tỉ -> tỷ (trừ "tỉ mỉ", "tỉ tê")
_OTO_RE = re.compile(r"\b([Ôô])tô\b")  # VnExpress viết "ôtô", báo khác viết "ô tô"
_EKIP_RE = re.compile(r"\b([Êê])[\s-]kíp\b")  # "ê-kíp" / "ê kíp" -> "êkíp"
_CTRL_RE = re.compile(r"[\u0000-\u001f\u007f-\u009f\u200b-\u200f\u2028\u2029\ufeff]")  # ký tự điều khiển, vô hình
_SPACE_RE = re.compile(r"\s+")
_HAS_LETTER_RE = re.compile(r"[^\W\d_]")  # có ít nhất 1 chữ cái (loại dấu câu, số)


def load_stopwords(path=STOPWORDS_TXT) -> set[str]:
    """Đọc stopwords: mỗi dòng một từ, dạng đã tách từ (cho_biết). Bỏ dòng trống và dòng bắt đầu bằng #."""
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if word and not word.startswith("#"):
                words.add(word)
    return words


def clean_text(text: str) -> str:
    """Bỏ HTML, URL, email, ký tự thừa; chuẩn hóa Unicode NFC, TP.HCM -> TP_HCM, tỉ -> tỷ; gộp khoảng trắng. Giữ hoa/thường."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = _CTRL_RE.sub(" ", text)
    # mỗi báo viết một kiểu: thống nhất để cùng một từ không bị tách thành nhiều từ khác nhau
    text = _TPHCM_RE.sub("TP_HCM", text)
    text = _TY_RE.sub(lambda m: "Tỷ" if m.group(0)[0] == "T" else "tỷ", text)
    text = _OTO_RE.sub(lambda m: m.group(1) + " tô", text)
    text = _EKIP_RE.sub(lambda m: m.group(1) + "kíp", text)
    return _SPACE_RE.sub(" ", text).strip()


def segment(text: str) -> str:
    """Tách từ tiếng Việt bằng pyvi: 'Việt Nam' -> 'Việt_Nam'."""
    return ViTokenizer.tokenize(text)


_EDGE_PUNCT = ".,;:!?\"'“”‘’()[]{}-–—…"


def make_text_tok(text: str, stopwords: set[str]) -> str:
    """clean_text -> tách từ -> chữ thường -> bỏ stopwords, dấu câu, số, token 1 ký tự.

    Tách từ TRƯỚC khi chuyển chữ thường, vì pyvi nhận ra tên riêng (Việt_Nam, Hà_Nội) tốt hơn
    khi còn chữ hoa. Các từ cách nhau 1 dấu cách.
    """
    tokens = (t.strip(_EDGE_PUNCT) for t in segment(clean_text(text)).lower().split())
    keep = [t for t in tokens if len(t) > 1 and _HAS_LETTER_RE.search(t) and t not in stopwords]
    return " ".join(keep)


def make_text_bert(title: str, sapo: str) -> str:
    """clean_text(title + '. ' + sapo). Không có sapo thì chỉ lấy tiêu đề."""
    title, sapo = clean_text(title), clean_text(sapo)
    if not sapo:
        return title
    return clean_text(f"{title}. {sapo}")


# Dòng thừa ở thân bài: bộ đếm bình luận, nút chia sẻ, link gợi ý, chú thích ảnh...
_JUNK_LINE_RE = re.compile(
    r"^(\(\s*\d*\s*\)|bình luận|chia sẻ|(xem thêm|đọc thêm|tin liên quan|mời độc giả)\b.*"
    r"|(ảnh|video|clip|nguồn)\s*:.*)$", re.I)
_CAPTION_END_RE = re.compile(r"\(\s*Ảnh\s*:[^)]*\)\s*\.?\s*$", re.I)  # '... (Ảnh: Nam Anh).'
_INLINE_CREDIT_RE = re.compile(r"\(\s*Ảnh\s*:[^)]*\)", re.I)
_SENTENCE_END = (".", "!", "?", "…", '"', "”", "’", "'", ":", ";")


def _clean_content(content: str) -> str:
    """Bỏ dòng thừa trong thân bài (bình luận, chú thích ảnh, tên tác giả ở cuối)."""
    paras = [clean_text(p) for p in str(content).split("\n")]
    paras = [p for p in paras if p and not _JUNK_LINE_RE.match(p) and not _CAPTION_END_RE.search(p)]
    paras = [_INLINE_CREDIT_RE.sub("", p).strip() for p in paras]
    # dòng cuối ngắn và không kết thúc bằng dấu câu = tên tác giả/nguồn ("Huy Đức tổng hợp")
    for _ in range(2):
        if paras and len(paras[-1].split()) <= 12 and not paras[-1].endswith(_SENTENCE_END):
            paras.pop()
    return "\n".join(paras)


def _full_text(title: str, sapo: str, content: str) -> str:
    """Toàn văn để tạo text_tok. Nhiều bài đã lặp sapo trong thân bài (thường ở đầu, đôi khi sau 1-2 câu dẫn,
    ~32 bài như vậy): có sapo trong thân bài thì không ghép lần nữa."""
    parts = [title]
    if sapo and clean_text(sapo)[:40] not in content:
        parts.append(sapo)
    parts.append(content)
    return ". ".join(parts)


def _read_raw(raw_dir) -> pd.DataFrame:
    """Đọc tất cả file *.jsonl trong raw_dir, bỏ dòng hỏng."""
    rows = []
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"Không có file .jsonl nào trong {raw_dir}. Chạy `python -m src.crawl` trước.")
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows)


def _title_key(title: str) -> str:
    """Khóa so sánh tiêu đề trùng: chữ thường, bỏ dấu câu và khoảng trắng thừa."""
    return _SPACE_RE.sub(" ", re.sub(r"[^\w\s]", " ", clean_text(title).lower())).strip()


def _pick_balanced(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Chọn n bài của 1 chủ đề, lấy đều giữa các báo (lần lượt mỗi báo 1 bài)."""
    queues = {s: g.sample(frac=1, random_state=seed).index.tolist()
              for s, g in group.groupby("source")}
    chosen = []
    while len(chosen) < n and any(queues.values()):
        for s in list(queues):
            if queues[s] and len(chosen) < n:
                chosen.append(queues[s].pop())
    return group.loc[chosen]


# Trang không phải bài báo (chính sách, điều khoản...) hoặc ngày giả (2010-01-01) lọt vào khi crawl
_NON_ARTICLE_TITLE_RE = re.compile(
    r"^(?:chính sách (?:bảo mật|quyền riêng tư)|điều khoản (?:sử dụng|dịch vụ)|quy chế hoạt động|liên hệ quảng cáo)\b", re.I)
MIN_YEAR = 2020


def _is_article(df: pd.DataFrame) -> pd.Series:
    """True với dòng là bài báo thật: tiêu đề không phải tên trang chính sách, ngày (nếu có) không quá cũ."""
    years = pd.to_numeric(df["date"].str[:4], errors="coerce")
    return ~df["title"].str.contains(_NON_ARTICLE_TITLE_RE) & ~(years < MIN_YEAR)


def _drop_near_duplicates(df: pd.DataFrame, threshold: float = NEAR_DUP_THRESHOLD) -> pd.DataFrame:
    """Loại bài gần trùng: cosine TF-IDF của text_tok > threshold (cùng một tin do các báo đăng lại).

    Duyệt theo thứ tự hiện có, giữ bài đầu tiên của mỗi nhóm trùng. Gộp cả các chủ đề với nhau
    vì một tin xuất hiện ở hai chuyên mục cũng làm nhãn bị nhiễu.
    """
    if len(df) < 2:
        return df
    tfidf = TfidfVectorizer(**VECTORIZER_PARAMS).fit_transform(df["text_tok"])
    sim = (tfidf @ tfidf.T).tocoo()  # tfidf đã chuẩn hóa L2 nên tích vô hướng = cosine
    dropped = set()
    for i, j, v in sorted(zip(sim.row, sim.col, sim.data)):
        if i < j and v > threshold and i not in dropped and j not in dropped:
            dropped.add(j)
            print(f"  gần trùng ({v:.3f}): giữ {df['url'].iloc[i]} | bỏ {df['url'].iloc[j]}")
    return df.iloc[[k for k in range(len(df)) if k not in dropped]]


def build_articles(raw_dir=DATA_RAW, seed=SEED, per_topic=None) -> pd.DataFrame:
    """Đọc *.jsonl -> lọc -> loại trùng và gần trùng -> cân bằng -> tạo 3 cột văn bản -> xáo trộn -> đánh id.

    per_topic: số bài mỗi chủ đề muốn lấy. Mặc định ARTICLES_PER_TOPIC; nếu có chủ đề không đủ
    thì hạ xuống bằng chủ đề ít nhất để các chủ đề cân bằng. Loại bài trước khi cân bằng, nên bài
    thừa (dự phòng) tự động bù vào chỗ bài bị loại.
    """
    df = _read_raw(raw_dir)
    n0 = len(df)
    df = df[df["true_label"].isin(TOPICS)].copy()
    for col in ["title", "sapo", "content", "date"]:
        df[col] = df[col].fillna("").astype(str)

    df["content"] = df["content"].map(_clean_content)
    # lọc: tiêu đề trống hoặc quá ngắn (<= 2 từ, thường là lấy nhầm), bài dưới MIN_WORDS từ, trang không phải bài báo
    df = df[(df["title"].str.split().str.len() > 2) & (df["content"].str.split().str.len() >= MIN_WORDS)]
    n_page = int((~_is_article(df)).sum())
    df = df[_is_article(df)]
    # loại trùng theo URL, rồi theo tiêu đề
    df = df.drop_duplicates("url")
    df["_tkey"] = df["title"].map(_title_key)
    df = df.drop_duplicates("_tkey").drop(columns="_tkey")
    print(f"Đọc {n0} bài -> sau lọc và loại trùng còn {len(df)} bài (trong đó đã bỏ {n_page} trang không phải bài báo)")

    # text_tok của toàn bộ bài còn lại, để so gần trùng trước khi cân bằng
    stopwords = load_stopwords()
    full = [_full_text(t, s, c) for t, s, c in zip(df["title"], df["sapo"], df["content"])]  # toàn văn
    df["text_tok"] = [make_text_tok(x, stopwords) for x in full]
    n1 = len(df)
    df = _drop_near_duplicates(df.reset_index(drop=True))
    print(f"Loại {n1 - len(df)} bài gần trùng (cosine TF-IDF > {NEAR_DUP_THRESHOLD}) -> còn {len(df)} bài")

    # cân bằng số bài giữa các chủ đề
    counts = df["true_label"].value_counts().reindex(list(TOPICS), fill_value=0)
    print("Số bài mỗi chủ đề (trước cân bằng):", counts.to_dict())
    target = min(per_topic or ARTICLES_PER_TOPIC, int(counts.min()))
    if target < MIN_ARTICLES_PER_TOPIC and per_topic is None:
        print(f"CẢNH BÁO: chủ đề ít nhất chỉ có {target} bài (< {MIN_ARTICLES_PER_TOPIC}). Cần crawl thêm.")
    elif per_topic is None and target < ARTICLES_PER_TOPIC:
        print(f"LƯU Ý: chủ đề ít nhất chỉ có {target} bài (< {ARTICLES_PER_TOPIC}), mỗi chủ đề lấy {target} bài.")
    parts = [_pick_balanced(g, target, seed + i) for i, (_, g) in enumerate(df.groupby("true_label"))]
    df = pd.concat(parts)

    # xáo trộn, đánh id (không lộ nhãn)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["id"] = [f"doc_{i + 1:04d}" for i in range(len(df))]

    # 2 phiên bản còn lại (text_tok đã tạo ở trên)
    df["text_bert"] = [make_text_bert(t, s) for t, s in zip(df["title"], df["sapo"])]
    df["text_bert_seg"] = df["text_bert"].map(segment)
    return df[ARTICLE_COLUMNS]


def load_articles(sample: bool = False) -> pd.DataFrame:
    """Đọc SAMPLE_CSV (sample=True) hoặc ARTICLES_CSV. Mọi người dùng hàm này để đọc dữ liệu."""
    path = SAMPLE_CSV if sample else ARTICLES_CSV
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Chạy `python -m src.preprocess"
                                f"{' --sample' if sample else ''}` hoặc `git pull` để lấy file mới.")
    return pd.read_csv(path, encoding="utf-8").fillna("")


def get_tokens(df: pd.DataFrame) -> list[list[str]]:
    """Danh sách token của từng bài, dùng cho LDA và coherence."""
    return df["text_tok"].str.split().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu")
    parser.add_argument("--sample", action="store_true",
                        help=f"tạo articles_sample.csv (~{SAMPLE_PER_TOPIC} bài/chủ đề) thay vì articles.csv")
    args = parser.parse_args()

    if args.sample:
        df = build_articles(per_topic=SAMPLE_PER_TOPIC)
        SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SAMPLE_CSV, index=False, encoding="utf-8")
        print(f"Đã lưu {len(df)} bài -> {SAMPLE_CSV}")
    else:
        df = build_articles()
        ARTICLES_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ARTICLES_CSV, index=False, encoding="utf-8")
        print(f"Đã lưu {len(df)} bài -> {ARTICLES_CSV}")
    print(df["true_label"].value_counts().to_string())
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
