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
TensorFlow v1 API 경고 : 불러온 Tensorflow 코드 중에 아직 구버전의 코드, 
                         오래되서 최신 버전엔 더 이상 사용하지 않는 코드로 인해 자잘한 경고문이 뜸
