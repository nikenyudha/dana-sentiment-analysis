import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# --- 1. SETTING HALAMAN ---
st.set_page_config(page_title="DANA Sentiment Analysis", layout="wide")
st.title("📊 Customer Sentiment & Topic Analysis")
st.markdown("""
Dashboard ini menganalisis review user aplikasi **DANA** menggunakan model **IndoBERT** yang telah di-fine-tune. 
Proyek ini membantu mengidentifikasi kepuasan pengguna dan area yang perlu diperbaiki.
""")

# --- 2. LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    model_path = "nikenlarash22/indobert-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

# Menggunakan path yang aman agar tidak error di server
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, 'data', 'reviews_cleaned.csv')
df = pd.read_csv(csv_path)

# Mapping Label (Hanya dilakukan sekali di sini)
df['Sentiment_Label'] = df['label'].map({1: 'Positif 😊', 0: 'Negatif 😡'})

# --- 3. SIDEBAR: UJI REAL-TIME ---
st.sidebar.header("🔍 Uji Model Real-Time")
st.sidebar.info("Ketik review di bawah untuk melihat bagaimana model AI mengklasifikasikannya.")
user_input = st.sidebar.text_area("Masukkan teks review:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        # Mendapatkan Probabilitas (Confidence Score)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=1).item()
        conf_score = torch.max(probs).item() * 100
    
    label = "POSITIF 😊" if prediction == 1 else "NEGATIF 😡"
    st.sidebar.markdown(f"### Hasil: **{label}**")
    st.sidebar.progress(conf_score / 100)
    st.sidebar.write(f"Tingkat Keyakinan Model: {conf_score:.2f}%")

# --- 4. MAIN DASHBOARD: VISUALISASI ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📈 Ringkasan Sentimen")
    sentiment_count = df['Sentiment_Label'].value_counts()
    st.bar_chart(sentiment_count)
    
    # Tambahan: Statistik Sederhana
    total_data = len(df)
    pos_perc = (df['label'] == 1).sum() / total_data * 100
    st.metric("Total Review", total_data)
    st.metric("Sentimen Positif", f"{pos_perc:.1f}%")

with col2:
    st.subheader("📄 Sampel Data Terbaru")
    st.dataframe(
        df[['content_cleaned', 'Sentiment_Label']].head(15),
        use_container_width=True
    )



st.markdown(
    "<hr style='margin-top:50px;'>"
    "<center style='color: gray;'>© 2026 Niken Larasati — Dana Customer Sentiment and Topic Analysis💗</center>",
    unsafe_allow_html=True
)


