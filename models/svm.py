import pandas as pd
import ast
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("data/10kreviews.csv", encoding="utf-8")

def safe_convert(text):
    try:
        return " ".join(ast.literal_eval(text))
    except (SyntaxError, ValueError):
        return str(text)

if "lemmatized_text" in df.columns:
    df["text"] = df["text"].apply(safe_convert)

X_text = df["text"]
y = df["stars"]

vectorizer = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{2,}\b",
    stop_words="english"
)
X_tfidf = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
