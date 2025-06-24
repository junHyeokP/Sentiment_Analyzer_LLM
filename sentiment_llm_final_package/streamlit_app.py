import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import load_stopwords, clean_and_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# 모델 및 토크나이저 로드
model = load_model("models/final_bilstm_attention.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
stopwords = load_stopwords("data/korean_stopwords.txt")
label_map = {0: '부정', 1: '중립', 2: '긍정'}

st.title("LLM 통합 감성분석 웹앱")

uploaded_file = st.file_uploader("CSV 파일 업로드 (컬럼명: document)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"총 {len(df)}개 문장 업로드됨")
    df['tokens'] = df['document'].apply(lambda x: clean_and_tokenize(x, stopwords))
    sequences = tokenizer.texts_to_sequences(df['tokens'])
    padded = pad_sequences(sequences, maxlen=100, padding='post')
    preds = model.predict(padded)
    pred_labels = [label_map[np.argmax(p)] for p in preds]
    df['예측'] = pred_labels

    st.subheader("감성분석 결과 분포")
    counts = Counter(pred_labels)
    st.bar_chart(pd.DataFrame(counts.values(), index=counts.keys(), columns=['count']))

    fig, ax = plt.subplots()
    plt.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
    st.pyplot(fig)

    st.subheader("키워드 분석 (WordCloud)")
    words = []
    for tokens in df['tokens']:
        words.extend(tokens)
    wc = WordCloud(font_path='NanumGothic.ttf', background_color='white', width=800, height=400).generate(' '.join(words))
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
