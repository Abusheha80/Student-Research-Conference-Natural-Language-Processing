import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------------------------------
# 1) LOAD DATA
# -------------------------------------------------------
df = pd.read_csv("data/100kreviews.csv")  # Path to your CSV file

# -------------------------------------------------------
# 2) PREPROCESS: MAP STARS TO SENTIMENT
# -------------------------------------------------------
#  1/2 stars -> negative
#  3 star    -> neutral
#  4/5 stars -> positive
def map_stars_to_sentiment(star):
    if star in [1, 2]:
        return "negative"
    elif star == 3:
        return "neutral"
    else:  # 4 or 5
        return "positive"

df["sentiment"] = df["stars"].apply(map_stars_to_sentiment)

# (Optional) If rows have missing values in key columns, drop them:
df.dropna(subset=["review", "stars"], inplace=True)

# -------------------------------------------------------
# 3) SPLIT DATA: 70% TRAIN / 30% TEST
# -------------------------------------------------------
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# -------------------------------------------------------
# 4) BUILD A PIPELINE WITH TF-IDF + RANDOM FOREST
# -------------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# -------------------------------------------------------
# 5) 5-FOLD CROSS VALIDATION ON TRAINING SET
# -------------------------------------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

print("Cross-validation accuracy scores (5-fold):", cv_scores)
print("Average cross-validation accuracy:", cv_scores.mean())

# -------------------------------------------------------
# 6) TRAIN FINAL MODEL ON THE ENTIRE TRAINING SET
# -------------------------------------------------------
pipeline.fit(X_train, y_train)

# -------------------------------------------------------
# 7) EVALUATE ON TEST SET
# -------------------------------------------------------
y_pred = pipeline.predict(X_test)
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_pred))

# -------------------------------------------------------
# 8) CONFUSION MATRIX & SAVE TO OUTPUT
# -------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])
print("Confusion Matrix:\n", cm)

# Create an output folder if it doesn't exist
output_folder = "data/matrix"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Plot the confusion matrix with matplotlib
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap=plt.cm.Blues)

# Set ticks and labels
labels = ["negative", "neutral", "positive"]
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

# Label the values in the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# Save the confusion matrix figure
matrix_path = os.path.join(output_folder, "rf_matrix.png")
plt.savefig(matrix_path)
plt.close()

print(f"Confusion matrix figure saved to: {matrix_path}")
