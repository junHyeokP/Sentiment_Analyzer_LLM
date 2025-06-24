import pickle
import numpy as np
from tqdm import tqdm
from gensim.models import FastText
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dropout, Bidirectional, LSTM, Dense, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.regularizers import l2

# 전처리된 데이터 로드
df = pickle.load(open("data/preprocessed.pkl", "rb"))

# FastText 임베딩 학습
fast = FastText(df['tokens'], vector_size=300, window=5, min_count=1, workers=4, epochs=15)
emb_dim = 300
vocab_size = 10000

# 토크나이저 생성
tok = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tok.fit_on_texts(df['tokens'])

# 임베딩 매트릭스 생성
embedding_matrix = np.zeros((vocab_size, emb_dim))
for word, i in tok.word_index.items():
    if i < vocab_size and word in fast.wv:
        embedding_matrix[i] = fast.wv[word]

# 학습 데이터 분할
X = df['tokens'].tolist()
y = df['label_enc'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

pad = lambda txts: pad_sequences(tok.texts_to_sequences(txts), maxlen=100, padding='post', truncating='post')
X_train_pad, X_test_pad = pad(X_train), pad(X_test)

# 모델 구성
inp = Input(shape=(100,))
x = Embedding(vocab_size, emb_dim, weights=[embedding_matrix], input_length=100, trainable=True)(inp)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.5)(x)
attn_out = Attention()([x, x])
x = tf.reduce_sum(attn_out, axis=1)
x = Dense(64, activation='relu')(x)
out = Dense(3, activation='softmax')(x)
model = Model(inp, out)
model.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 학습
model.fit(X_train_pad, y_train, epochs=20, batch_size=64, validation_split=0.1,
          callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                     ModelCheckpoint("models/final_bilstm_attention.h5", save_best_only=True)])

# 토크나이저 저장
import pickle
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tok, f)

print("학습 및 저장 완료")
