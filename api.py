from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import uvicorn

app = FastAPI()

# =========================
# 🔹 LOAD DATASETS
# =========================

fake = pd.read_csv("Fake.csv").sample(500, random_state=42)
true = pd.read_csv("True.csv").sample(500, random_state=42)
ag = pd.read_csv("train.csv").sample(500, random_state=42)
welfake = pd.read_csv("WELFake_Dataset.csv").sample(500, random_state=42)
huff = pd.read_json("News_Category_Dataset_v3.json", lines=True).sample(500, random_state=42)

# =========================
# 🔹 PROCESS DATA
# =========================

# Fake & True
fake["label"] = 0
true["label"] = 1
fake = fake[["text", "label"]]
true = true[["text", "label"]]

# AG News
ag["text"] = ag["Title"] + " " + ag["Description"]
ag["label"] = 1
ag = ag[["text", "label"]]

# WELFake
welfake = welfake[["text", "label"]]

huff["text"] = huff["headline"] + " " + huff["short_description"]
huff["label"] = 1
huff = huff[["text","label"]]
# =========================
# 🔹 COMBINE ALL
# =========================

df = pd.concat([fake, true, ag, welfake, huff])
df = df.dropna()

# Balance classes
min_count = df["label"].value_counts().min()

fake_news = df[df["label"] == 0].sample(min_count, random_state=42)
real_news = df[df["label"] == 1].sample(min_count, random_state=42)

df = pd.concat([fake_news, real_news]).sample(frac=1).reset_index(drop=True)

# =========================
# 🔹 TRAIN MODEL
# =========================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)
model.fit(X, y)

print("✅ Model trained successfully")

# =========================
# 🔹 API ROUTES
# =========================

@app.get("/")
def home():
    return {"message": "Fake News ML API Running 🚀"}

@app.get("/predict")
def predict(news: str):
    vec = vectorizer.transform([news])

    probability = model.predict_proba(vec)[0]

    # probability[1] = REAL probability
    real_prob = probability[1]

    if real_prob >= 0.65:
        prediction = "REAL"
        confidence_score = real_prob
    else:
        prediction = "FAKE"
        confidence_score = 1 - real_prob

    return {
        "prediction": prediction,
        "confidence": round(float(confidence_score), 3)
    }

# =========================
# 🔹 RUN SERVER
# =========================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)