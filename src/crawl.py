"""Thu thập bài báo: 6 chủ đề x 4 báo -> data/raw/<chu_de>.jsonl.

Chạy từ thư mục gốc repo:
    python -m src.crawl --test            # chạy thử 2 trang mỗi báo, KHÔNG ghi file
    python -m src.crawl --max-per-source 5  # lấy ít để làm dữ liệu mẫu (ngày 1)
    python -m src.crawl                   # chạy đầy đủ (~120 bài/báo/chủ đề), chạy tiếp được nếu bị ngắt
"""
import argparse
import json
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.config import DATA_RAW, MIN_WORDS, REQUEST_DELAY, SOURCES, TOPICS

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "vi,en;q=0.8",
}
MAX_RETRY = 3  # tối đa 3 lần thử lại

# Chuyên mục của từng báo (KE_HOACH §3.1): {chu_de: {bao: url trang chuyên mục}}
CATEGORY_URLS = {
    "thoi_su": {
        "vnexpress": "https://vnexpress.net/thoi-su",
        "tuoitre": "https://tuoitre.vn/thoi-su.htm",
        "thanhnien": "https://thanhnien.vn/thoi-su.htm",
        "dantri": "https://dantri.com.vn/xa-hoi.htm",
    },
    "kinh_te": {
        "vnexpress": "https://vnexpress.net/kinh-doanh",
        "tuoitre": "https://tuoitre.vn/kinh-doanh.htm",
        "thanhnien": "https://thanhnien.vn/kinh-te.htm",
        "dantri": "https://dantri.com.vn/kinh-doanh.htm",
    },
    "the_thao": {
        "vnexpress": "https://vnexpress.net/the-thao",
        "tuoitre": "https://tuoitre.vn/the-thao.htm",
        "thanhnien": "https://thanhnien.vn/the-thao.htm",
        "dantri": "https://dantri.com.vn/the-thao.htm",
    },
    "giai_tri": {
        "vnexpress": "https://vnexpress.net/giai-tri",
        "tuoitre": "https://tuoitre.vn/giai-tri.htm",
        "thanhnien": "https://thanhnien.vn/giai-tri.htm",
        "dantri": "https://dantri.com.vn/giai-tri.htm",
    },
    "cong_nghe": {
        "vnexpress": "https://vnexpress.net/khoa-hoc-cong-nghe",
        "tuoitre": "https://tuoitre.vn/cong-nghe.htm",
        "thanhnien": "https://thanhnien.vn/cong-nghe.htm",
        "dantri": "https://dantri.com.vn/cong-nghe.htm",
    },
    "suc_khoe": {
        "vnexpress": "https://vnexpress.net/suc-khoe",
        "tuoitre": "https://tuoitre.vn/suc-khoe.htm",
        "thanhnien": "https://thanhnien.vn/suc-khoe.htm",
        "dantri": "https://dantri.com.vn/suc-khoe.htm",
    },
}

# Mẫu URL của một BÀI BÁO (để lọc link menu, quảng cáo, trang chuyên mục)
ARTICLE_URL_RE = {
    "vnexpress": re.compile(r"^https?://(www\.)?vnexpress\.net/[^?#]+-\d{6,}\.html$"),
    "tuoitre": re.compile(r"^https?://(www\.)?tuoitre\.vn/[^?#]+-\d{10,}\.htm$"),
    "thanhnien": re.compile(r"^https?://(www\.)?thanhnien\.vn/[^?#]+-\d{10,}\.htm$"),
    "dantri": re.compile(r"^https?://(www\.)?dantri\.com\.vn/[^?#]+-\d{10,}\.htm$"),
}
# Bỏ bài video, ảnh, infographic, bài trực tiếp
SKIP_URL_PARTS = ("/video", "/photo", "/anh/", "/infographic", "/podcast", "/emagazine",
                  "/truc-tiep", "/long-form")

# Vùng chứa thân bài: thử lần lượt từng selector, cái nào có nhiều đoạn <p> nhất thì dùng
CONTENT_SELECTORS = {
    "vnexpress": ["article.fck_detail", "article.content-detail", "div.fck_detail"],
    "tuoitre": ["div.detail-content", "div.detail-cmain", "div#main-detail-body"],
    "thanhnien": ["div.detail-content", "div.details__content", "div.cms-body"],
    "dantri": ["div.singular-content", "div.e-magazine__body", "div.detail-content"],
}
TITLE_SELECTORS = ["h1.title-detail", "h1.detail-title", "h1.article-title",
                   "h1.title-page", "h1.details__headline"]
SAPO_SELECTORS = ["p.description", "h2.detail-sapo", "div.detail-sapo", "h2.singular-sapo",
                  "div.sapo", "h2.sapo", "h2.details__summary"]
# Phần cần bỏ khỏi thân bài (ảnh, chú thích, tin liên quan, quảng cáo)
NOISE_SELECTORS = ["script", "style", "figure", "figcaption", "table", "iframe",
                   "[class*=related]", "[class*=VCSortableInPreviewMode]", "[class*=box-news]",
                   "[class*=banner]", "[class*=advertise]", "[class*=author]"]


def _fetch(url: str) -> str | None:
    """Tải 1 trang. Lỗi mạng hoặc HTTP 429/5xx: chờ tăng dần (2s, 4s, 8s) rồi thử lại."""
    for attempt in range(MAX_RETRY + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                time.sleep(REQUEST_DELAY)
                resp.encoding = "utf-8"
                return resp.text
            if resp.status_code not in (429, 500, 502, 503, 504):
                time.sleep(REQUEST_DELAY)
                return None  # 404, 403...: không thử lại
        except requests.RequestException:
            pass
        if attempt < MAX_RETRY:
            time.sleep(2 ** (attempt + 1))
    return None


def _page_url(source: str, base: str, page: int) -> str:
    """URL trang chuyên mục thứ `page` (trang 1 là URL gốc). Quy tắc phân trang từng báo."""
    if page == 1:
        return base
    if source == "vnexpress":                       # /the-thao -> /the-thao-p2
        return f"{base}-p{page}"
    if source == "tuoitre":                         # /the-thao.htm -> /the-thao/trang-2.htm
        return base.replace(".htm", f"/trang-{page}.htm")
    if source == "thanhnien":                       # /the-thao.htm -> /the-thao-trang-2.htm
        return base.replace(".htm", f"-trang-{page}.htm")
    if source == "dantri":                          # /the-thao.htm -> /the-thao/trang-2.htm
        return base.replace(".htm", f"/trang-{page}.htm")
    raise ValueError(f"Báo không hợp lệ: {source}")


def _category_slug(url: str) -> str:
    """'https://tuoitre.vn/the-thao.htm' -> 'the-thao'."""
    return urlparse(url).path.strip("/").split("/")[0].replace(".htm", "")


def _clean_url(href: str, page_url: str) -> str:
    return urljoin(page_url, href).split("#")[0].split("?")[0]


def list_article_urls(source: str, topic: str, max_pages: int = 10) -> list[str]:
    """Duyệt trang chuyên mục 1 -> max_pages, trả về danh sách link bài (không trùng, giữ thứ tự)."""
    base = CATEGORY_URLS[topic][source]
    pattern = ARTICLE_URL_RE[source]
    seen, urls = set(), []
    for page in range(1, max_pages + 1):
        page_url = _page_url(source, base, page)
        html = _fetch(page_url)
        if html is None:
            break  # hết trang hoặc lỗi
        soup = BeautifulSoup(html, "lxml")
        new = 0
        for a in soup.find_all("a", href=True):
            link = _clean_url(a["href"], page_url)
            if link in seen or not pattern.match(link):
                continue
            if any(part in link for part in SKIP_URL_PARTS):
                continue
            seen.add(link)
            urls.append(link)
            new += 1
        if new == 0:
            break  # trang này không có link mới: dừng, tránh lặp vô hạn
    return urls


def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    for sel in selectors:
        tag = soup.select_one(sel)
        if tag and tag.get_text(strip=True):
            return tag.get_text(" ", strip=True)
    return ""


_SAPO_PREFIX_RE = re.compile(r"^\(?(Dân trí|Tuổi Trẻ|TTO|Thanh Niên|VnExpress)\)?\s*[-–—:]\s*", re.I)


def _extract_sapo(soup: BeautifulSoup) -> str:
    """Lấy sapo, bỏ tên địa điểm ở đầu (VnExpress: 'Anh', 'Hungary') và tiền tố '(Dân trí) -'."""
    text = ""
    for sel in SAPO_SELECTORS:
        tag = soup.select_one(sel)
        if tag is None:
            continue
        for junk in tag.select(".location-stamp, [class*=location]"):
            junk.decompose()
        text = tag.get_text(" ", strip=True)
        if text:
            break
    if not text:
        text = _meta(soup, "og:description")
    return _SAPO_PREFIX_RE.sub("", text).strip()


def _meta(soup: BeautifulSoup, key: str) -> str:
    tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
    return tag["content"].strip() if tag and tag.get("content") else ""


def _parse_date(soup: BeautifulSoup) -> str:
    """Trả về ngày dạng YYYY-MM-DD, không đọc được thì trả chuỗi rỗng."""
    candidates = [_meta(soup, "article:published_time"), _meta(soup, "pubdate"),
                  _meta(soup, "datePublished")]
    for script in soup.find_all("script", type="application/ld+json"):
        m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', script.get_text())
        if m:
            candidates.append(m.group(1))
    time_tag = soup.find("time")
    if time_tag:
        candidates.append(time_tag.get("datetime", "") or time_tag.get_text(" ", strip=True))
    for tag in soup.select("span.date, div.detail-time, .date-time"):
        candidates.append(tag.get_text(" ", strip=True))
    for text in candidates:
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
        if m:
            return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}"
    return ""


def _breadcrumb_ok(soup: BeautifulSoup, topic: str, source: str) -> bool:
    """Đối chiếu breadcrumb để chắc bài thuộc đúng chuyên mục.

    Trang không có breadcrumb thì KHÔNG nhận: bài báo thật luôn có breadcrumb, còn trang
    chính sách, điều khoản, liên hệ... thì không (ví dụ "Chính sách bảo mật dữ liệu cá nhân" của Dân trí).
    """
    slug = _category_slug(CATEGORY_URLS[topic][source])
    crumbs = soup.select("ul.breadcrumb a, .breadcrumb a, [class*=breadcrumb] a, "
                         "[class*=bread-crumb] a")
    if not crumbs:
        return False
    for a in crumbs:
        for seg in urlparse(a.get("href", "")).path.strip("/").split("/"):
            if seg.replace(".htm", "") == slug:
                return True
    return False


def _extract_content(soup: BeautifulSoup, source: str) -> str:
    """Lấy thân bài: gộp các đoạn <p> trong vùng bài viết, bỏ ảnh/chú thích/tin liên quan."""
    best = []
    for sel in CONTENT_SELECTORS[source] + ["article"]:
        box = soup.select_one(sel)
        if box is None:
            continue
        for noise in box.select(",".join(NOISE_SELECTORS)):
            noise.decompose()
        paras = [p.get_text(" ", strip=True) for p in box.find_all("p")]
        paras = [p for p in paras if len(p.split()) >= 3]
        if len(paras) > len(best):
            best = paras
    return "\n".join(best)


def parse_article(url: str, source: str, topic: str) -> dict | None:
    """Tải 1 bài và trích title, sapo, content, url, source, date, true_label.

    Trả về None nếu lỗi tải, sai chuyên mục, hoặc bài không hợp lệ (video/ảnh, dưới MIN_WORDS từ).
    """
    html = _fetch(url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "lxml")
    if not _breadcrumb_ok(soup, topic, source):
        return None
    # h1 riêng của báo -> og:title -> h1 bất kỳ (h1 chung có thể là tên mục, ví dụ "Tin tức")
    title = _first_text(soup, TITLE_SELECTORS) or _meta(soup, "og:title") or _first_text(soup, ["h1"])
    sapo = _extract_sapo(soup)
    content = _extract_content(soup, source)
    if not title or len(content.split()) < MIN_WORDS:  # bỏ bài video/ảnh, bài quá ngắn
        return None
    return {
        "title": title,
        "sapo": sapo,
        "content": content,
        "url": url,
        "source": source,
        "date": _parse_date(soup),
        "true_label": topic,
    }


def _load_done(path) -> tuple[set[str], dict[str, int]]:
    """Đọc file jsonl đã có: tập URL đã tải và số bài đã có của mỗi báo (để chạy tiếp)."""
    urls, per_source = set(), {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # dòng hỏng do bị ngắt giữa chừng
                urls.add(rec["url"])
                per_source[rec["source"]] = per_source.get(rec["source"], 0) + 1
    return urls, per_source


def crawl_topic(topic: str, max_per_source: int = 120) -> None:
    """Thu thập 1 chủ đề từ các báo, ghi nối tiếp vào DATA_RAW/<topic>.jsonl, bỏ qua URL đã có."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW / f"{topic}.jsonl"
    done_urls, per_source = _load_done(path)
    for source in SOURCES:
        have = per_source.get(source, 0)
        if have >= max_per_source:
            print(f"[{topic}/{source}] đã đủ {have} bài, bỏ qua")
            continue
        urls = [u for u in list_article_urls(source, topic) if u not in done_urls]
        print(f"[{topic}/{source}] có {have} bài, tìm thấy {len(urls)} link mới")
        for url in urls:
            if have >= max_per_source:
                break
            rec = parse_article(url, source, topic)
            done_urls.add(url)  # nhớ link đã thử trong lần chạy này (link lỗi không ghi file nên lần sau thử lại)
            if rec is None:
                continue
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            have += 1
            if have % 10 == 0:
                print(f"[{topic}/{source}] {have}/{max_per_source}")
        print(f"[{topic}/{source}] xong: {have} bài")


def _run_test(topics: list[str]) -> None:
    """Chạy thử: mỗi báo 2 trang, đọc 3 bài, in kết quả để kiểm tra bằng mắt. Không ghi file."""
    for topic in topics:
        for source in SOURCES:
            print(f"\n=== {topic} / {source} ===")
            urls = list_article_urls(source, topic, max_pages=2)
            print(f"Số link tìm được (2 trang): {len(urls)}")
            if not urls:
                print("  -> KHÔNG lấy được link: kiểm tra CATEGORY_URLS, _page_url, ARTICLE_URL_RE")
                continue
            ok = 0
            for url in urls[:3]:
                rec = parse_article(url, source, topic)
                if rec is None:
                    print(f"  [None] {url}")
                    continue
                ok += 1
                print(f"  [OK] {url}")
                print(f"       title : {rec['title'][:80]}")
                print(f"       sapo  : {rec['sapo'][:80]}")
                print(f"       date  : {rec['date']}   số từ nội dung: {len(rec['content'].split())}")
            print(f"Đọc được {ok}/3 bài")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thu thập bài báo")
    parser.add_argument("--test", action="store_true", help="chạy thử 2 trang mỗi báo, không ghi file")
    parser.add_argument("--topic", choices=list(TOPICS), help="chỉ chạy 1 chủ đề")
    parser.add_argument("--max-per-source", type=int, default=120, help="số bài tối đa mỗi báo mỗi chủ đề")
    args = parser.parse_args()
    topics = [args.topic] if args.topic else list(TOPICS)
    if args.test:
        _run_test(topics)
        return
    for topic in topics:
        crawl_topic(topic, max_per_source=args.max_per_source)


if __name__ == "__main__":
    main()
