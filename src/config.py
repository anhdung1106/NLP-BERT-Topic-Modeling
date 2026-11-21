from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Đường dẫn
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTICLES_CSV = DATA_PROCESSED / "articles.csv"
SAMPLE_CSV = DATA_PROCESSED / "articles_sample.csv"
STOPWORDS_TXT = ROOT / "data" / "stopwords" / "vietnamese_stopwords.txt"
EMB_DIR = ROOT / "embeddings"
MODELS_DIR = ROOT / "models"
LDA_DIR = MODELS_DIR / "lda"
BERTOPIC_DIR = MODELS_DIR / "bertopic_best"
TABLES_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
RESULTS_CSV = TABLES_DIR / "results.csv"
DOC_TOPICS_CSV = TABLES_DIR / "doc_topics.csv"
TOPIC_WORDS_CSV = TABLES_DIR / "topic_words.csv"
EMB_TIME_CSV = TABLES_DIR / "embedding_time.csv"

# Chung
SEED = 42
SEEDS = [42, 43, 44]
N_TOPICS = 6
TOP_N_WORDS = 10
NOISE_LABEL = -1

# Dữ liệu
TOPICS = {
    "thoi_su": "Thời sự",
    "kinh_te": "Kinh tế",
    "the_thao": "Thể thao",
    "giai_tri": "Giải trí",
    "cong_nghe": "Công nghệ",
    "suc_khoe": "Sức khỏe",
}
SOURCES = ["vnexpress", "tuoitre", "thanhnien", "dantri"]
ARTICLES_PER_TOPIC = 300
MIN_ARTICLES_PER_TOPIC = 200
MIN_WORDS = 150
SAMPLE_PER_TOPIC = 20  # số bài mỗi chủ đề trong articles_sample.csv
NEAR_DUP_THRESHOLD = 0.95  # cosine TF-IDF của text_tok: lớn hơn mức này coi là cùng một bài
REQUEST_DELAY = 1.0
ARTICLE_COLUMNS = ["id", "title", "sapo", "content", "url", "source", "date",
                   "true_label", "text_tok", "text_bert", "text_bert_seg"]

# Vectorizer: text_tok đã chữ thường, tách từ, các từ cách nhau 1 dấu cách
TOKEN_PARAMS = dict(token_pattern=r"(?u)\S+", lowercase=False, ngram_range=(1, 1))
VECTORIZER_PARAMS = dict(**TOKEN_PARAMS, min_df=3, max_df=0.5)  # LSA, B0, từ khóa cụm
BERTOPIC_VECTORIZER_PARAMS = dict(**TOKEN_PARAMS)               # c-TF-IDF của BERTopic

# Mô hình truyền thống
SVD_DIMS = [10, 20, 50]
LDA_PASSES = 10

# Embedding
EMBEDDINGS = {
    "b0_tfidf_svd":  {"model": None, "pooling": None, "text_col": "text_tok", "dim": 100},
    "b1_mbert_cls":  {"model": "bert-base-multilingual-cased", "pooling": "cls", "text_col": "text_bert"},
    "b2_mbert_mean": {"model": "bert-base-multilingual-cased", "pooling": "mean", "text_col": "text_bert"},
    "b3_minilm":     {"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "pooling": "sbert", "text_col": "text_bert"},
    "b4_vi_sbert":   {"model": "keepitreal/vietnamese-sbert", "pooling": "sbert", "text_col": "text_bert_seg"},
    "b5_e5":         {"model": "intfloat/multilingual-e5-base", "pooling": "sbert", "text_col": "text_bert", "prefix": "query: "},
}
MAX_LENGTH = 256

# BERTopic
UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
MIN_CLUSTER_SIZES = [15, 30, 50]
DEFAULT_MIN_CLUSTER_SIZE = 30

# Cấu hình chốt: giá trị tạm, TV4 cập nhật sau ngày 4
BEST_EMBEDDING = "b3_minilm"
BEST_SVD_DIM = 50
BEST_MIN_CLUSTER_SIZE = 30
