import pandas as pd
import pickle
from utils import load_stopwords, clean_and_tokenize

# 불용어 불러오기
stopwords = load_stopwords("data/korean_stopwords.txt")

# 데이터 로드
df = pd.read_csv("data/sample_data.csv")

# 라벨 인코딩 (부정:0, 중립:1, 긍정:2)
label_map = {"부정": 0, "중립": 1, "긍정": 2}
df = df[df['label'].isin(label_map.keys())].copy()
df['label_enc'] = df['label'].map(label_map)

# 토큰화
df['tokens'] = df['document'].apply(lambda x: clean_and_tokenize(x, stopwords))

# 저장
pickle.dump(df, open("data/preprocessed.pkl", "wb"))
print(f"전처리 완료: {len(df)}건 저장")
