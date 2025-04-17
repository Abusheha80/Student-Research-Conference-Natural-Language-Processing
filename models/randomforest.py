import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

df = pd.read_csv('data/1mreviews.csv')

def map_sentiment(star):
    if star in [1, 2]:
        return 0
    elif star == 3:
        return 1
    else:
        return 2

df['label'] = df['stars'].apply(map_sentiment)
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

#random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

#metrics
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

outdir = 'output/matrix'
os.makedirs(outdir, exist_ok=True)

#plot confusion matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{outdir}/random_forest_confusion_matrix.png')
plt.close()

#plot roc curve
plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve((y_test==i).astype(int), y_prob[:, i])
    plt.plot(fpr, tpr, label=f'Class {i}')
plt.title('Random Forest ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(f'{outdir}/random_forest_roc_curve.png')
plt.close()

df_metrics = pd.DataFrame({
    'Accuracy': [acc],
    'Precision': [prec],
    'Recall': [rec],
    'F1 Score': [f1],
    'AUC': [auc],
    'CV Mean Accuracy': [cv_scores.mean()]
})
df_metrics.to_csv(f'{outdir}/random_forest_metrics.csv', index=False)
