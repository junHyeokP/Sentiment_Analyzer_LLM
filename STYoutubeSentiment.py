import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter
from crawler.youtube_crawler import collect_youtube_comments
from preprocess.text_cleaner import clean_text
from model.predict import predict_labels
from wordcloud import WordCloud
import matplotlib.font_manager as fm

# Windows 시스템에서 맑은 고딕 지정
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rcParams["font.family"] = font_name


st.title("🎯 실시간 유튜브 감성분석 (KoBERT)")

query = st.text_input("🔍 인물 이름을 검색하고 유튜브 댓글로 담겨진 인물에 대한 평가를 분석해보세요.")

if st.button("댓글 수집 및 감성분석"):
    if not query.strip():
        st.warning("검색어를 입력하세요.")
    else:
        st.info(f"🔎 '{query}' 관련 유튜브 댓글 수집 중...")
        raw_comments = collect_youtube_comments(query)
        st.success(f"{len(raw_comments)}개 댓글 수집 완료!")

        st.info("🧼 전처리 및 예측 중...")
        cleaned = [clean_text(c) for c in raw_comments if len(c.strip()) > 1]
        pred_labels = predict_labels(cleaned)

        df = pd.DataFrame({"document": cleaned, "label": pred_labels})
        os.makedirs("data", exist_ok=True)
        df.to_csv("./data/new_data.csv", index=False, encoding="utf-8-sig")
        st.success("✅ 감성 분석 완료 및 new_data.csv 저장!")

        st.subheader("감성 분포 - Bar Chart")
        counts = Counter(pred_labels)
        st.bar_chart(
            pd.DataFrame(counts.values(),
            index=counts.keys(), 
            columns=["Count"])
            )

        st.subheader("감성 분포 - Pie Chart")
        fig, ax = plt.subplots()
        plt.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%")
        st.pyplot(fig)

        st.subheader("WordCloud")
    

        wc = WordCloud(
            font_path="./hangul/NanumGothic-Regular.ttf",
            background_color="white",   
            width=800,
            height=400
        )

        wc.generate(" ".join(cleaned))

        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)