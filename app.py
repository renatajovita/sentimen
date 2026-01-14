import streamlit as st
import pandas as pd
import re, emoji
import joblib
import random

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(
    page_title="Sentiment Analysis GUI (Illustration Mode)",
    layout="wide"
)

# ======================================================
# LOAD SLANG DICTIONARY (TETAP SAMA)
# ======================================================
@st.cache_resource
def load_slang_dict(path="assets/combined_slang_word.txt"):
    slang_dict = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    slang, formal = parts
                    slang_dict[slang.strip()] = formal.strip()
    except FileNotFoundError:
        pass
    return slang_dict

slang_dict = load_slang_dict()

# ======================================================
# PREPROCESSING (IDENTIK DENGAN FINAL)
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
# LOAD LABEL ENCODER (OPTIONAL)
# ======================================================
@st.cache_resource
def load_label_encoder():
    try:
        return joblib.load("label_encoder_sentiment.pkl")
    except:
        return ["negative", "neutral", "positive"]

lbl = load_label_encoder()

# ======================================================
# DUMMY MODEL LOADER (ILUSTRASI)
# ======================================================
@st.cache_resource
def load_model(model_name):
    # HANYA ILUSTRASI
    return f"Tokenizer({model_name})", f"Model({model_name})"

tokenizer_bert, model_bert = load_model("IndoBERT (Illustration)")
tokenizer_deb, model_deb   = load_model("DeBERTa (Illustration)")

# ======================================================
# DUMMY PREDICTION FUNCTION (FLOW SAMA)
# ======================================================
def predict_sentiment(texts, tokenizer, model):
    """
    Ini simulasi output model.
    Logika sederhana + deterministik supaya tutorial konsisten.
    """
    results = []

    for text in texts:
        if any(word in text for word in ["bagus", "mantap", "suka", "senang", "puas"]):
            results.append("positive")
        elif any(word in text for word in ["jelek", "lama", "kecewa", "parah", "buruk"]):
            results.append("negative")
        else:
            results.append("neutral")

    return results

# ======================================================
# UI
# ======================================================
st.title("📊 Sentiment Analysis (Illustration Mode)")
st.info(
    "⚠️ **Mode Ilustrasi** — Model belum di-load. "
    "Output hanya untuk kebutuhan tutorial & visualisasi alur sistem."
)

st.markdown(
    """
Aplikasi analisis sentimen menggunakan **IndoBERT-base-p1** dan **DeBERTa-v3-base**  
(dalam mode ilustrasi).  
Flow sistem **identik** dengan versi final.
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

            bert_pred = predict_sentiment(
                [clean_text],
                tokenizer_bert,
                model_bert
            )[0]

            deb_pred = predict_sentiment(
                [clean_text],
                tokenizer_deb,
                model_deb
            )[0]

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

            with st.spinner("⏳ Melakukan prediksi sentimen (simulasi)..."):
                df["sentimen_bert"] = predict_sentiment(
                    df["clean_text"].tolist(),
                    tokenizer_bert,
                    model_bert
                )
                df["sentimen_deberta"] = predict_sentiment(
                    df["clean_text"].tolist(),
                    tokenizer_deb,
                    model_deb
                )

            output_df = df[["Text", "sentimen_bert", "sentimen_deberta"]]

            st.subheader("✅ Hasil Prediksi")
            st.dataframe(output_df, use_container_width=True)

            csv = output_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Hasil (CSV)",
                csv,
                "hasil_sentimen_ilustrasi.csv",
                "text/csv",
                use_container_width=True
            )
