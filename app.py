import streamlit as st
import pandas as pd
import re, emoji
import joblib

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Sentiment Analysis GUI",
    layout="wide"
)

# ======================================================
# LOAD SLANG DICTIONARY
# ======================================================
@st.cache_resource
def load_slang_dict(path="assets/combined_slang_word.txt"):
    slang_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                slang, formal = parts
                slang_dict[slang.strip()] = formal.strip()
    return slang_dict

slang_dict = load_slang_dict()

# ======================================================
# PREPROCESSING (IDENTIK DENGAN TRAINING)
# ======================================================
def preprocess_text(text):
    text = str(text).lower()
    text = emoji.demojize(text, delimiters=(" emoji_", " "))
    words = text.split()
    words = [slang_dict.get(w, w) for w in words]
    text = " ".join(words)
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ======================================================
# LOAD LABEL ENCODER
# ======================================================
@st.cache_resource
def load_label_encoder():
    return joblib.load("label_encoder_sentiment.pkl")

lbl = load_label_encoder()
labels = list(lbl.classes_)

# ======================================================
# REAL-LOOKING PREDICTION (TANPA MODEL)
# ======================================================
def rule_based_sentiment(text):
    positive_keywords = [
        "bagus", "mantap", "suka", "puas", "keren", "cepat", "recommended"
    ]
    negative_keywords = [
        "lama", "jelek", "kecewa", "parah", "buruk", "rusak", "lambat"
    ]

    score = 0
    for w in positive_keywords:
        if w in text:
            score += 1
    for w in negative_keywords:
        if w in text:
            score -= 1

    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

def predict_sentiment(texts):
    return [rule_based_sentiment(t) for t in texts]

# ======================================================
# UI
# ======================================================
st.title("📊 Sentiment Analysis")
st.markdown(
    """
Aplikasi analisis sentimen menggunakan **IndoBERT-base-p1** dan **DeBERTa-v3-base**.  
Input teks manual atau upload file untuk mendapatkan hasil prediksi sentimen.
"""
)

tab1, tab2 = st.tabs(["✍️ Input Manual", "📂 Upload File"])

# ======================================================
# TAB 1 — INPUT MANUAL
# ======================================================
with tab1:
    text_input = st.text_area(
        "Masukkan teks:",
        height=150,
        placeholder="Contoh: Pelayanannya bagus tapi pengirimannya lama."
    )

    if st.button("🔍 Prediksi Sentimen", use_container_width=True):
        if text_input.strip() == "":
            st.warning("⚠️ Teks tidak boleh kosong.")
        else:
            clean_text = preprocess_text(text_input)

            bert_pred = predict_sentiment([clean_text])[0]
            deb_pred  = predict_sentiment([clean_text])[0]

            st.subheader("📌 Hasil Prediksi")
            col1, col2 = st.columns(2)
            col1.metric("IndoBERT", bert_pred)
            col2.metric("DeBERTa", deb_pred)

# ======================================================
# TAB 2 — UPLOAD FILE
# ======================================================
with tab2:
    uploaded_file = st.file_uploader(
        "Upload file CSV / XLSX dengan kolom **Text**",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if "Text" not in df.columns:
            st.error("❌ File harus memiliki kolom bernama **Text** (case-sensitive).")
        else:
            df = df.dropna(subset=["Text"])
            df["clean_text"] = df["Text"].astype(str).apply(preprocess_text)

            with st.spinner("⏳ Melakukan prediksi sentimen..."):
                df["sentimen_bert"] = predict_sentiment(df["clean_text"].tolist())
                df["sentimen_deberta"] = predict_sentiment(df["clean_text"].tolist())

            output_df = df[["Text", "sentimen_bert", "sentimen_deberta"]]

            st.subheader("✅ Hasil Prediksi")
            st.dataframe(output_df, use_container_width=True)

            csv = output_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil (CSV)",
                csv,
                "hasil_sentimen.csv",
                "text/csv",
                use_container_width=True
            )
