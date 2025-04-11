import pandas as pd
import numpy as np
import tensorflow as tf
# Fix the keras import paths
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, 
    Dense, Dropout, concatenate
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load and preprocess data
df = pd.read_csv("data/100kreviews.csv")

def map_sentiment(stars):
    if stars in [1, 2]:
        return "negative"
    elif stars == 3:
        return "neutral"
    else:
        return "positive"

df["sentiment"] = df["stars"].apply(map_sentiment)
X = df["text"]
y = df["sentiment"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Tokenization
max_words = 5000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# CNN model
model = Sequential()
model.add(Embedding(max_words, 64, input_length=max_len))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_seq, y_train_cat, epochs=5, batch_size=32, verbose=1)

# Evaluate
y_pred = model.predict(X_test_seq)
y_pred_labels = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

os.makedirs("output/matrix", exist_ok=True)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("CNN Confusion Matrix")
plt.savefig("output/matrix/cnn_confusion_matrix.png")
plt.close()

print("Accuracy:", accuracy_score(y_test, y_pred_labels))
print(classification_report(y_test, y_pred_labels, target_names=le.classes_))
