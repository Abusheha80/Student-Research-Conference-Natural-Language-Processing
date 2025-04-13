import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Load data
df = pd.read_csv('data/100kreviews.csv')

# Map stars to sentiment
def map_sentiment(star):
    if star in [1, 2]:
        return 0  # negative
    elif star == 3:
        return 1  # neutral
    else:
        return 2  # positive

df['label'] = df['stars'].apply(map_sentiment)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['text'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Model and 5-fold CV
model = MultinomialNB()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
y_bin = label_binarize(y_test, classes=[0,1,2])
auc = roc_auc_score(y_bin, y_prob, multi_class='ovr')

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"CV Mean Accuracy: {cv_scores.mean():.4f}")

# Confusion Matrix and ROC
outdir = 'output/matrix'
os.makedirs(outdir, exist_ok=True)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Naive Bayes Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{outdir}/naive_bayes_confusion_matrix.png')
plt.close()

plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
    plt.plot(fpr, tpr, label=f'Class {i}')
plt.title('Naive Bayes ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/naive_bayes_roc_curve.png')
plt.close()

# Save metrics
df_metrics = pd.DataFrame({
    'Accuracy': [acc],
    'Precision': [prec],
    'Recall': [rec],
    'F1 Score': [f1],
    'AUC': [auc],
    'CV Mean Accuracy': [cv_scores.mean()]
})
df_metrics.to_csv(f'{outdir}/naive_bayes_metrics.csv', index=False)
