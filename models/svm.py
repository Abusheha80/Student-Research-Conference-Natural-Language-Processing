import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv('data/1mreviews.csv')  # change path if needed

# Map stars to sentiment
def map_sentiment(star):
    if star in [1, 2]:
        return 0
    elif star == 3:
        return 1
    else:
        return 2

df['label'] = df['stars'].apply(map_sentiment)

# Vectorize
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['text'])
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# 5-fold CV
svm = SVC(probability=True, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring='accuracy')

# Train
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# AUC
# For multi-class, use one-vs-rest
from sklearn.preprocessing import label_binarize
y_binarized = label_binarize(y_test, classes=[0,1,2])
auc = roc_auc_score(y_binarized, y_prob, multi_class='ovr')

# Output folder
output_dir = 'output/matrix'
os.makedirs(output_dir, exist_ok=True)

# Confusion matrix plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{output_dir}/svm_confusion_matrix.png')
plt.close()

# ROC plot
plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve((y_test==i).astype(int), y_prob[:, i])
    plt.plot(fpr, tpr, label=f'Class {i}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/svm_roc_curve.png')
plt.close()

# Print
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"CV Mean Accuracy: {np.mean(cv_scores):.4f}")

# Save metrics
metrics_df = pd.DataFrame({
    'Accuracy': [acc],
    'Precision': [prec],
    'Recall': [rec],
    'F1 Score': [f1],
    'AUC': [auc],
    'CV Mean Accuracy': [np.mean(cv_scores)]
})
metrics_df.to_csv(f'{output_dir}/svm_metrics.csv', index=False)
