import pandas as pd
import ast
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1) Load your dataset (filtered or unfiltered—but be consistent)
#    e.g., a CSV with a "text" column (already lemmatized) and a "stars" column (1–5).
df = pd.read_csv("reviews.csv", encoding="utf-8")

# 2) (Optional) Convert stringified token lists to text if needed
def safe_convert(text):
    try:
        return " ".join(ast.literal_eval(text))
    except (SyntaxError, ValueError):
        return str(text)

if "lemmatized_text" in df.columns:
    df["text"] = df["lemmatized_text"].apply(safe_convert)

# 3) (Optional) Filter non-English rows if not already done
def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

df = df[df["text"].apply(is_english)]

# 4) Split into features (X_text) and labels (y)
X_text = df["text"]
y = df["stars"]  # or whatever your target column is

# 5) TF-IDF vectorization (in-memory, no .npz file)
vectorizer = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{2,}\b",
    stop_words="english"
)
X_tfidf = vectorizer.fit_transform(X_text)

# 6) Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7) Train an SVM classifier
#    For text classification, a linear kernel often performs well,
#    but feel free to experiment with other kernels.
svm = SVC(kernel="linear", random_state=42)
svm.fit(X_train, y_train)


# 8) Evaluate
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
