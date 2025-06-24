import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import load_stopwords, clean_and_tokenize

# 모델 & 토크나이저 로드
model = load_model("models/final_bilstm_attention.h5")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

stopwords = load_stopwords("data/korean_stopwords.txt")

# 추론 함수
def predict_sentiment(sentence):
    tokens = clean_and_tokenize(sentence, stopwords)
    seq = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(padded)
    label_map = {0: '부정', 1: '중립', 2: '긍정'}
    return label_map[np.argmax(pred)], pred

# 예시
example = "정말 서비스가 훌륭했어요"
label, prob = predict_sentiment(example)
print(f"문장: {example}")
print(f"예측: {label} (확률분포: {prob})")
