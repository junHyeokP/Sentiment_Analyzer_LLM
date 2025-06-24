import re
import pickle
from konlpy.tag import Okt

# 불용어 로더
def load_stopwords(path="data/korean_stopwords.txt"):
    with open(path, encoding='utf-8') as f:
        stopwords = set(f.read().splitlines())
    return stopwords

# 전처리 + 토큰화
def clean_and_tokenize(text, stopwords):
    okt = Okt()
    text = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s!?]", "", str(text))
    tokens = okt.morphs(text, stem=True)
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

# 토크나이저 저장 & 로드
def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
