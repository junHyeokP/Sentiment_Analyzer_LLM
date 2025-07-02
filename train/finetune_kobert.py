from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers.models.bert import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import os, numpy as np, re
from torch.nn import CrossEntropyLoss

# ✅ 설정
MODEL_DIR = "models"
LABEL_MAP = {"부정": 0, "중립": 1, "긍정": 2}
TOKENIZER_NAME = "monologg/kobert"


# ✅ 가장 최신 모델 버전 찾기
def get_latest_model_version():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        return 0
    pattern = re.compile(r"kobert_v(\d+)\.pt")
    versions = [int(pattern.search(f).group(1)) for f in os.listdir(MODEL_DIR) if pattern.search(f)]
    return max(versions) if versions else 0

# ✅ 커스텀 데이터셋
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    df = pd.read_csv("../data/new_data.csv")
    print(df["label"].value_counts())
    df = df[df['label'].isin(LABEL_MAP)]
    df['label_enc'] = df['label'].map(LABEL_MAP)
    print(df["label_enc"].value_counts())

  
weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    #loss_fn = CrossEntropyLoss(weight=class_weights)

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)
    model = BertForSequenceClassification.from_pretrained(TOKENIZER_NAME, num_labels=3)

    # 최신 버전 모델 로드
    version = get_latest_model_version()
    latest_model_path = os.path.join(MODEL_DIR, f"kobert_v{version}.pt")
    if os.path.exists(latest_model_path):
        print(f"🔁 기존 모델 로딩: {latest_model_path}")
        model.load_state_dict(torch.load(latest_model_path, map_location="cpu"))

    # 데이터
    dataset = SentimentDataset(df['document'].tolist(), df['label_enc'].tolist(), tokenizer)
   

    X_train, X_val, y_train, y_val = train_test_split(df['document'], df['label_enc'], test_size=0.1, stratify=df['label_enc'])

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0,1,2]),
        y=np.array(y_train)
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    train_dataset = SentimentDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    val_dataset = SentimentDataset(X_val.tolist(), y_val.tolist(), tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)


    # 학습
    optimizer = AdamW(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(2):  # 전이학습은 짧게
        loop = tqdm(train_loader, desc=f"[Fine-tune Epoch {epoch+1}]")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch).logits
            loss = loss_fn(logits, batch["labels"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

# ✅ 성능 평가

    epoch_acc_list = []

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels)

    print("\n📊 [검증 성능 보고서]")
    acc = accuracy_score(y_true, y_pred)
    epoch_acc_list.append(acc)
    print(f"✅ Epoch {epoch+1} 검증 정확도: {acc:.4f}")



# 저장
with open(f"{MODEL_DIR}/eval_report_v{next_version}.txt", "w", encoding="utf-8") as f:
    f.write(report)


    # 다음 버전으로 저장
    next_version = version + 1
    save_path = os.path.join(MODEL_DIR, f"kobert_v{next_version}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"✅ 모델 저장 완료: {save_path}")

    # 모델도 kobert_latest.pt로 복사
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "kobert_latest.pt"))
