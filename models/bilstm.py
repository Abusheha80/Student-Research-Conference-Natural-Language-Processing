import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, 
    Dense, Dropout
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_csv("data/reviews.csv", encoding="utf-8")
texts = df["text"].astype(str).tolist()
labels = df["stars"].values

X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

vocab_size = 20000  
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq   = tokenizer.texts_to_sequences(X_val)

max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_val_pad   = pad_sequences(X_val_seq,   maxlen=max_len, padding='post')

y_train = y_train - 1
y_val   = y_val   - 1

embedding_dim = 128

input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(
    input_dim=vocab_size, 
    output_dim=embedding_dim, 
    input_length=max_len
)(input_layer)

bilstm = Bidirectional(LSTM(128, return_sequences=False))(embedding_layer)
dropout = Dropout(0.5)(bilstm)
output = Dense(5, activation='softmax')(dropout)

model = Model(inputs=input_layer, outputs=output)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=5,
    batch_size=64
)

pred_probs = model.predict(X_val_pad)
y_pred = np.argmax(pred_probs, axis=1)

print("Classification Report:")
print(classification_report(y_val, y_pred))
