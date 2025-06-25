## Youtube Comment Sentimental Analyzer

# 구성


# 이용방법
![image](https://github.com/user-attachments/assets/ea4b8e78-5424-484c-b6e2-f0d0e4491bd1)

검색 바에서 요즘 인식이 궁금한 인물 이름을 적고 버튼을 누르면 바로 감성분석이 진행됩니다.

![image](https://github.com/user-attachments/assets/87599075-f9e9-4a39-bae6-712c81e7e401)

검색시 웹크롤링이 바로 진행되어 영상 소리가 들리니 주의,

감성분석 결과는 bar chart, pychart로 긍정/중립/부정 비율이 나타나며, 

wordcloud로 댓글 중 키워드 빈도수가 가장 많은 단어들을 표시합니다.

# 코드 작성 중 발견한 문제점

TensorFlow v1 API 경고 : 불러온 Tensorflow 코드 중에 아직 구버전의 코드가 존재 
                         
                         오래되서 최신 버전엔 더 이상 사용하지 않는 코드로 인해 자잘한 경고문이 뜸
                         
                         성능엔 별 영향이 없을 것으로 판단되어 현재 고치고 있진 않지만, 코드 리펙토링이 필요하다고 인지는 하는 중임.

저장된 모델 파일 의존성 : 모델(상대 경로 : model/kobert_latest.pt)이 없으면 웹 안으로도 못 들어가며 사실상 아무 기능도 동작을 못함
                 
                 상대 경로 형식대로 모델이 놓여있지 않다면,
                 
                 전처리된 csv파일을 이용하여 모델 학습을 하여 해당 경로로 파일을 저장해야 함.
