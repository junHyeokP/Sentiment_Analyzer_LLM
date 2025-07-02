import re

KOREAN_STOPWORDS = set([
    "이", "그", "저", "것", "수", "들", "등", "는", "은", "가", "을", "를",
    "에", "의", "가장", "좀", "좀더", "너무", "정말", "진짜", "그리고",
    "해서", "하지만", "그러나", "또한", "되다", "있다", "없다"
])

def clean_text(text):
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", str(text))
    words = text.strip().split()
    return " ".join([w for w in words if w not in KOREAN_STOPWORDS and len(w) > 1])

