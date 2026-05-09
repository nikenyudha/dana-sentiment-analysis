import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- 1. SETTING HALAMAN ---
st.set_page_config(page_title="E-Wallet Sentiment Dashboard", layout="wide")
st.title("📊 Customer Sentiment & Topic Analysis")
st.markdown("Dashboard ini menganalisis review user aplikasi **DANA** menggunakan Deep Learning (IndoBERT).")

# --- 2. LOAD MODEL & DATA ---
@st.cache_resource # Agar model tidak di-load ulang setiap klik
def load_model():
    model_path = "nikenlarash22/indobert-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()
df = pd.read_csv('./data/reviews_cleaned.csv')

# --- 3. SIDEBAR: INPUT TEKS BARU ---
st.sidebar.header("Uji Kalimat Baru")
user_input = st.sidebar.text_area("Masukkan review:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    label = "POSITIF 😊" if prediction == 1 else "NEGATIF 😡"
    st.sidebar.subheader(f"Hasil: {label}")

# --- 4. MAIN DASHBOARD: STATISTIK ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Ringkasan Sentimen")
    sentiment_count = df['label'].value_counts()
    st.bar_chart(sentiment_count)

with col2:
    st.subheader("Sampel Data Review")
    st.dataframe(df[['content_cleaned', 'label']].head(10))

st.success("Saran: Gunakan data dari BERTopic untuk membuat chart topik di sini!")