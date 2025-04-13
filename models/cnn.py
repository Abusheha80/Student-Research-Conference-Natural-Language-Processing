import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
df = pd.read_csv('data/100kreviews.csv')

# Map stars to sentiment
def map_sentiment(star):
    if star in [1, 2]:
        return 0
    elif star == 3:
        return 1
    else:
        return 2

df['label'] = df['stars'].apply(map_sentiment)

# Prepare data
texts = df['text'].astype(str).values
labels = df['label'].values
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, stratify=labels, random_state=42
)

# Convert text to sequences
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LEN = 200
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train_texts)

X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
X_test_seq = tokenizer.texts_to_sequences(X_test_texts)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LEN, padding='post', truncating='post')

# Build CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_VOCAB_SIZE, output_dim=100, input_length=MAX_SEQUENCE_LEN))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 5-fold cross validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []
for train_idx, val_idx in skf.split(X_train_pad, y_train):
    cnn_cv = create_cnn_model()
    X_train_cv, X_val_cv = X_train_pad[train_idx], X_train_pad[val_idx]
    y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
    cnn_cv.fit(X_train_cv, y_train_cv, epochs=3, batch_size=128, 
               validation_data=(X_val_cv, y_val_cv), verbose=0)
    _, val_acc = cnn_cv.evaluate(X_val_cv, y_val_cv, verbose=0)
    cv_accuracies.append(val_acc)

# Train final model
model = create_cnn_model()
model.fit(X_train_pad, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

# Predict
y_pred_probs = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probs, axis=1)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)
y_test_bin = tf.keras.utils.to_categorical(y_test, num_classes=3)
auc = roc_auc_score(y_test_bin, y_pred_probs, multi_class='ovr')

# Print
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"CV Mean Accuracy: {np.mean(cv_accuracies):.4f}")

# Confusion Matrix and ROC
outdir = 'output/matrix'
os.makedirs(outdir, exist_ok=True)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CNN Confusion Matrix')
plt.tight_layout()
plt.savefig(f'{outdir}/cnn_confusion_matrix.png')
plt.close()

plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    plt.plot(fpr, tpr, label=f'Class {i}')
plt.title('CNN ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(f'{outdir}/cnn_roc_curve.png')
plt.close()

# Save metrics
df_metrics = pd.DataFrame({
    'Accuracy': [acc],
    'Precision': [prec],
    'Recall': [rec],
    'F1 Score': [f1],
    'AUC': [auc],
    'CV Mean Accuracy': [np.mean(cv_accuracies)]
})
df_metrics.to_csv(f'{outdir}/cnn_metrics.csv', index=False)
