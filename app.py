import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import re

st.set_page_config(page_title="Sentiment Analysis - Raja Ampat", layout="wide")

st.title("📊 Analisis Sentimen: Tambang di Raja Ampat")
st.markdown("Aplikasi ini menganalisis sentimen masyarakat terhadap isu pertambangan di Raja Ampat menggunakan data dari Twitter.")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload file CSV tweet:", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'\@\w+|\#','', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text

    df["Clean_Tweet"] = df["Tweet"].astype(str).apply(clean_text)

    # Sentiment
    def get_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"

    df["Sentiment"] = df["Clean_Tweet"].apply(get_sentiment)

    st.subheader("🔎 Sampel Data")
    st.dataframe(df.head())

    st.subheader("📊 Distribusi Sentimen")
    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    fig1, ax1 = plt.subplots()
    sentiment_counts.plot.pie(autopct="%1.1f%%", startangle=90, ax=ax1, colors=['red', 'gray', 'green'])
    ax1.set_ylabel('')
    st.pyplot(fig1)

    st.subheader("☁️ Word Cloud")
    all_words = ' '.join(df["Clean_Tweet"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(wordcloud, interpolation='bilinear')
    ax2.axis("off")
    st.pyplot(fig2)

    st.success("✅ Analisis selesai. Anda dapat mengganti file CSV untuk analisis lainnya.")
else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
