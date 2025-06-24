# LLM 통합 감성분석 프로젝트

## 프로젝트 구성

- 전처리: `preprocess.py`
- 학습: `train_model.py`
- 추론: `inference.py`
- Streamlit 웹앱: `streamlit_app.py`
- 유틸리티: `utils.py`

## 실행 순서
```bash

1️⃣ 데이터 준비 (`data/sample_data.csv`)  

2️⃣ 전처리: (python) preprocess.py

3️⃣ 학습: (python) train_model.py

4️⃣ 웹앱 실행:

streamlit run streamlit_app.py
CSV 파일 포맷
document: 감성 분석 대상 문장

label: 부정 / 중립 / 긍정

불용어 사전
data/korean_stopwords.txt 수정 가능
