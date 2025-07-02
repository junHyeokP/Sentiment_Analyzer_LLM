## Youtube Comment Sentimental Analyzer

# 이용방법
![image](https://github.com/user-attachments/assets/ea4b8e78-5424-484c-b6e2-f0d0e4491bd1)

검색 바에서 요즘 인식이 궁금한 인물 이름을 적고 버튼을 누르면 바로 감성분석이 진행됩니다.

검색시 웹크롤링이 바로 진행되어 영상 소리가 들리니 주의,

감성분석 결과는 bar chart, pychart로 긍정/중립/부정 비율이 나타나며, 

wordcloud로 댓글 중 키워드 빈도수가 가장 많은 단어들을 표시합니다.

# 기술 스택

Python, Pandas, matplotlib(막대 그래프, 워드 클라우드), numpy
Google gemma(언어모델)
Selenium(웹크롤러)
sklearn

# 코드 작성 중 발견한 문제점 / 취약점

### TensorFlow v1 API 경고 : 불러온 Tensorflow 코드 중에 아직 구버전의 코드가 존재 
  
오래되서 최신 버전엔 더 이상 사용하지 않는 코드로 인해 자잘한 경고문이 뜸

성능엔 별 영향이 없을 것으로 판단되어 현재 고치고 있진 않지만, 코드 리펙토링이 필요하다고 인지는 하는 중임.
                        
### 저장된 모델 파일 의존성 : 모델(상대 경로 : models/kobert_latest.pt)이 없으면 웹 안으로도 
### 못 들어가며 사실상 아무 기능도 동작을 못함

상대 경로 형식대로 모델이 놓여있지 않다면,

전처리된 .csv 파일을 이용하여 모델 학습을 하여 해당 경로로 파일을 저장해야 함.

미리 저장된 .csv 파일을 쓸 것  (경로 : data/new_sentiment_0_387.csv)

## 추후 개선 예정

검색 결과가 '중립'에만 집중되어 집계되는 현상을 해결하기위해 데이터 수집을 계속진행

현재 사용중인 모델인 kobert모델을 위한 epoch, learning_rate 조정 및 점검


# 프로젝트를 하며 느낀 사항

VSCODE로 코드를 작성할때 VSCODE 기준의 경로 설정이 익숙치 않아서 테스트하는데 어려움을 유발했다.
