## 유튜브 댓글 감성분석 프로젝트

## 프로젝트 구성

- 전처리: `preprocess.py`
- 학습: `train_model.py`
- 추론: `inference.py`
- Streamlit 웹앱: `streamlit_app.py`
- 유틸리티: `utils.py'


웹앱 실행 방법 :

streamlit 실행 streamlit_app.py
CSV 파일 포맷
document: 감성 분석 대상 문장 

label: 부정 / 중립 / 긍정

불용어 사전
data/korean_stopwords.txt 수정 가능

양방향 LSTM 사용,
데이터를 좀더 수집하여 더 정확한 결과를 낼 예정
