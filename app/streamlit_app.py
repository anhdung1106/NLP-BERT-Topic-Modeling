"""Demo: Kiến trúc BERT và ứng dụng trong topic modeling (TV5).

Chạy từ thư mục gốc repo:  streamlit run app/streamlit_app.py
App chỉ đọc kết quả có sẵn trong outputs/, embeddings/, models/, không huấn luyện lại.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (BEST_EMBEDDING, BEST_MIN_CLUSTER_SIZE, BEST_SVD_DIM, DOC_TOPICS_CSV,
                        EMB_TIME_CSV, EMBEDDINGS, NOISE_LABEL, RESULTS_CSV, TOPIC_WORDS_CSV, TOPICS)
from src.preprocess import load_articles

st.set_page_config(page_title="BERT & Topic Modeling", layout="wide")

METHODS = {
    "bertopic_k6": "BERTopic (k = 6)",
    "bertopic_hdbscan": "BERTopic (HDBSCAN)",
    "bert_kmeans": "BERT + K-Means",
    "lsa_kmeans": "LSA + K-Means",
    "lda": "LDA",
}
EMB_NAMES = {"b0_tfidf_svd": "B0. TF-IDF + SVD", "b1_mbert_cls": "B1. mBERT [CLS]",
             "b2_mbert_mean": "B2. mBERT mean pooling", "b3_minilm": "B3. Multilingual MiniLM",
             "b4_vi_sbert": "B4. Vietnamese-SBERT", "b5_e5": "B5. Multilingual-E5"}
METRICS = {"nmi": "NMI", "ari": "ARI", "purity": "Purity", "cv": "C_v", "diversity": "Diversity"}
SAMPLES = [
    "Giá đường tăng mạnh do nguồn cung khan hiếm",
    "Đường cao tốc Bắc - Nam đoạn mới chính thức thông xe",
    "Đội tuyển Việt Nam thắng 2-0 ở vòng loại",
    "Bác sĩ cảnh báo bệnh sốt xuất huyết tăng mạnh",
    "Nvidia ra mắt chip AI mới cho trung tâm dữ liệu",
]


# ---------------------------------------------------------------- dữ liệu
@st.cache_data
def load_tables():
    df = load_articles()
    dt = pd.read_csv(DOC_TOPICS_CSV)
    res = pd.read_csv(RESULTS_CSV, keep_default_na=False)
    tw = pd.read_csv(TOPIC_WORDS_CSV)
    emb_time = pd.read_csv(EMB_TIME_CSV).set_index("embedding")["time_s"]
    return df, dt, res, tw, emb_time


@st.cache_data
def topic_names(method: str) -> dict:
    """topic_id -> 'Chủ đề t: 3 từ khóa (≈ chuyên mục chiếm đa số)'."""
    _, dt, _, tw, _ = load_tables()
    words = dict(zip(tw.loc[tw.method == method, "topic"], tw.loc[tw.method == method, "words"]))
    if not words:  # bert_kmeans chưa có trong topic_words.csv: tính từ text_tok
        from src.baselines import top_words_by_cluster
        df = load_tables()[0]
        ids = sorted(t for t in dt[method].unique() if t != NOISE_LABEL)
        lab = dt[method].values
        tops = top_words_by_cluster(df.loc[lab != NOISE_LABEL, "text_tok"].tolist(), lab[lab != NOISE_LABEL])
        words = {t: " ".join(w) for t, w in zip(ids, tops)}
    names = {}
    for t in sorted(dt[method].unique()):
        if t == NOISE_LABEL:
            names[t] = "Nhiễu (-1)"
            continue
        major = dt.loc[dt[method] == t, "true_label"].value_counts().index[0]
        kw = ", ".join(str(words.get(t, "")).split()[:3])
        names[t] = f"Chủ đề {t}: {kw} (≈ {TOPICS[major]})"
    return names


@st.cache_resource
def load_models():
    from src.baselines import load_lda
    from src.bertopic_runner import load_bertopic
    lda, dictionary = load_lda()
    return lda, dictionary, load_bertopic()


@st.cache_data
def best_embedding():
    from src.embed import load_embedding
    return load_embedding(BEST_EMBEDDING)


def mean_std(rows: pd.DataFrame, name: str) -> dict:
    out = {"Phương pháp": name}
    for k, v in METRICS.items():
        out[v] = f"{rows[k].mean():.3f} ± {rows[k].std():.3f}"
    out["Thời gian (s)"] = round(rows["time_s"].mean(), 1)
    return out


df, dt, res, tw, emb_time = load_tables()
st.title("Kiến trúc BERT và ứng dụng trong topic modeling")
st.caption(f"{len(df):,} bài báo tiếng Việt · 6 chuyên mục · embedding chốt: {EMB_NAMES[BEST_EMBEDDING]}")
tab1, tab2, tab3, tab4 = st.tabs(["1. Bản đồ chủ đề", "2. Khám phá chủ đề",
                                  "3. Phân tích văn bản mới", "4. So sánh mô hình"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    c1, c2 = st.columns([2, 1])
    method = c1.selectbox("Mô hình", list(METHODS), format_func=METHODS.get, key="map_method")
    by_label = c2.toggle("Tô màu theo nhãn thật (chuyên mục báo)")
    plot = dt[["x", "y", "true_label", method]].copy()
    plot["Tiêu đề"] = df["title"].values
    plot["Chuyên mục"] = plot["true_label"].map(TOPICS)
    plot["Chủ đề"] = plot[method].map(topic_names(method))
    color = "Chuyên mục" if by_label else "Chủ đề"
    fig = px.scatter(plot, x="x", y="y", color=color, hover_name="Tiêu đề",
                     hover_data={"x": False, "y": False, "Chủ đề": True, "Chuyên mục": True},
                     height=620)
    fig.update_traces(marker=dict(size=5, opacity=0.8))
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, legend_title=None)
    st.plotly_chart(fig, width="stretch")
    st.caption("Mỗi chấm là một bài báo, vị trí lấy từ UMAP 2 chiều của embedding E5. Rê chuột để xem tiêu đề.")

# ---------------------------------------------------------------- Tab 2
with tab2:
    method = st.selectbox("Mô hình", list(METHODS), format_func=METHODS.get, key="explore_method")
    names = topic_names(method)
    topic = st.selectbox("Chủ đề", list(names), format_func=names.get)
    mask = (dt[method] == topic).values
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("Số bài", int(mask.sum()))
        row = tw[(tw.method == method) & (tw.topic == topic)]
        if len(row):
            st.markdown("**Từ khóa:** " + ", ".join(row.iloc[0]["words"].split()))
        comp = dt.loc[mask, "true_label"].map(TOPICS).value_counts().rename_axis("Chuyên mục").reset_index(name="Số bài")
        st.dataframe(comp, hide_index=True, width="stretch")
    with c2:
        st.markdown("**5 bài tiêu biểu** (gần tâm chủ đề nhất)")
        X = best_embedding()
        center = X[mask].mean(axis=0)
        idx = np.where(mask)[0][np.argsort(-(X[mask] @ center))[:5]]
        for i in idx:
            st.markdown(f"- {df.loc[i, 'title']}  \n  <small>{df.loc[i, 'source']} · {TOPICS[df.loc[i, 'true_label']]}</small>",
                        unsafe_allow_html=True)

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.markdown("Câu mẫu (bấm để điền):")
    cols = st.columns(len(SAMPLES))
    for col, s in zip(cols, SAMPLES):
        if col.button(s, width="stretch"):
            st.session_state["new_text"] = s
    text = st.text_area("Nhập một đoạn tin tức", key="new_text", height=120)
    if st.button("Phân tích", type="primary") and text.strip():
        from src.baselines import predict_lda
        from src.bertopic_runner import predict_bertopic
        lda, dictionary, bt_model = load_models()
        with st.spinner("Đang tính embedding (lần đầu sẽ tải mô hình E5, mất vài phút)..."):
            t_lda, p_lda = predict_lda(text, lda, dictionary)
            t_bt, p_bt = predict_bertopic(text, bt_model)
        c1, c2 = st.columns(2)
        c1.subheader("LDA")
        c1.markdown(f"**{topic_names('lda')[t_lda]}**")
        c1.progress(float(p_lda), text=f"Xác suất: {p_lda:.0%}")
        c2.subheader("BERTopic (k = 6)")
        c2.markdown(f"**{topic_names('bertopic_k6')[t_bt]}**")
        c2.progress(float(p_bt), text=f"Mức tin cậy: {p_bt:.0%}")
        st.caption("LDA dựa trên đếm từ; BERTopic dựa trên embedding E5 (hiểu ngữ cảnh). "
                   "Mức tin cậy BERTopic = softmax độ tương đồng với tâm các chủ đề, không phải xác suất của mô hình.")

# ---------------------------------------------------------------- Tab 4
with tab4:
    sel = lambda m, e, mode, p="": res[(res.method == m) & (res.embedding == e) & (res["mode"] == mode) & (res.param == p)]
    t1 = pd.DataFrame([
        mean_std(sel("lda", "bow", "k6"), "A1. LDA"),
        mean_std(sel("lsa_kmeans", "tfidf_svd", "k6", f"svd_dim={BEST_SVD_DIM}"), "A2. LSA + K-Means"),
        mean_std(sel("bert_kmeans", BEST_EMBEDDING, "k6"), "A3. BERT + K-Means"),
        mean_std(sel("bertopic", BEST_EMBEDDING, "k6"), "A4. BERTopic (k = 6)"),
        mean_std(sel("bertopic", BEST_EMBEDDING, "hdbscan", f"min_cluster_size={BEST_MIN_CLUSTER_SIZE}"),
                 "A4. BERTopic (HDBSCAN)"),
    ])
    rows = []
    for e in EMBEDDINGS:
        k6, hd = sel("bertopic", e, "k6"), sel("bertopic", e, "hdbscan", f"min_cluster_size={BEST_MIN_CLUSTER_SIZE}")
        rows.append({"Mô hình": EMB_NAMES[e], "NMI (k=6)": k6.nmi.mean(), "ARI (k=6)": k6.ari.mean(),
                     "C_v (k=6)": k6.cv.mean(), "NMI (HDBSCAN)": hd.nmi.mean(),
                     "Số chủ đề (HDBSCAN)": hd.n_topics.mean(), "Tỉ lệ nhiễu": hd.pct_noise.mean(),
                     "Thời gian embedding (s)": emb_time[e]})
    t2 = pd.DataFrame(rows).round(3)

    metric = st.radio("Chỉ số", list(METRICS.values()), horizontal=True)
    st.subheader("Bảng 1 – BERT so với phương pháp truyền thống (TB 3 seed)")
    t1_num = t1.assign(v=t1[metric].str.split(" ± ").str[0].astype(float))
    st.plotly_chart(px.bar(t1_num, x="Phương pháp", y="v", text="v", labels={"v": metric}, height=360),
                    width="stretch")
    st.dataframe(t1, hide_index=True, width="stretch")
    st.subheader("Bảng 2 – Cách lấy biểu diễn trong BERTopic (TB 3 seed)")
    col = {"NMI": "NMI (k=6)", "ARI": "ARI (k=6)", "C_v": "C_v (k=6)"}.get(metric, "NMI (k=6)")
    st.plotly_chart(px.bar(t2, x="Mô hình", y=col, text=col, height=360), width="stretch")
    st.dataframe(t2, hide_index=True, width="stretch")
    st.caption(f"HDBSCAN dùng min_cluster_size = {BEST_MIN_CLUSTER_SIZE}. Bảng 2 có NMI, ARI, C_v; "
               "Purity và Diversity chỉ hiện ở Bảng 1.")
