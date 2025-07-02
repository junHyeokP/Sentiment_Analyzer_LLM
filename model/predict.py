import torch
from model.kobert_model import load_model, load_tokenizer, LABEL_MAP

MAX_LEN = 128
model = load_model()
tokenizer = load_tokenizer()

def predict_labels(texts):
    encoded = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)
        preds = torch.argmax(outputs.logits, dim=1).numpy()
    return [LABEL_MAP[p] for p in preds]