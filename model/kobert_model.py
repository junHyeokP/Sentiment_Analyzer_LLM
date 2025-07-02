import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "../train/models/kobert_latest.pt"
TOKENIZER_NAME = "monologg/kobert"

LABEL_MAP = {0: "부정", 1: "중립", 2: "긍정"}

@torch.no_grad()
def load_model():
    model = BertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=3)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    except FileNotFoundError:
        print(f"[경고] 모델 파일이 존재하지 않습니다: {MODEL_PATH}. 예측 및 웹 앱 기능이 제한됩니다.")
    model.eval()
    return model

def load_tokenizer():
    return BertTokenizer.from_pretrained(TOKENIZER_NAME)