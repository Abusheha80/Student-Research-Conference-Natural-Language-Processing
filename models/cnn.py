import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D, 
    Dense, Dropout, concatenate
)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1) LOAD YOUR DATA
df = pd.read_csv("data/10kreviews.csv", encoding="utf-8")
texts = df["text"].astype(str).tolist()
labels = df["stars"].values

# 2) TRAIN/TEST SPLIT
X_train, X_val, y_train, y_val = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# 3) TOKENIZE & CONVERT TEXT TO SEQUENCES
vocab_size = 20000  
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq   = tokenizer.texts_to_sequences(X_val)

# 4) PAD SEQUENCES
max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_val_pad   = pad_sequences(X_val_seq,   maxlen=max_len, padding='post')

# 5) SHIFT LABELS FROM [1..5] TO [0..4] IF NEEDED
y_train = y_train - 1
y_val   = y_val   - 1

# 6) BUILD A MULTI-CHANNEL CNN MODEL
embedding_dim = 128  # or 200/300 if you prefer

input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(
    input_dim=vocab_size, 
    output_dim=embedding_dim, 
    input_length=max_len
)(input_layer)

# --- PARALLEL CONV LAYERS WITH DIFFERENT KERNEL SIZES ---
conv_k3 = Conv1D(filters=128, kernel_size=3, activation='relu')(embedding_layer)
pool_k3 = GlobalMaxPooling1D()(conv_k3)

conv_k4 = Conv1D(filters=128, kernel_size=4, activation='relu')(embedding_layer)
pool_k4 = GlobalMaxPooling1D()(conv_k4)

conv_k5 = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding_layer)
pool_k5 = GlobalMaxPooling1D()(conv_k5)

# --- CONCATENATE THE POOLED FEATURES ---
concat = concatenate([pool_k3, pool_k4, pool_k5])

# --- DENSE LAYERS ---
dense = Dense(128, activation='relu')(concat)
dropout = Dropout(0.5)(dense)
output = Dense(5, activation='softmax')(dropout)

model = Model(inputs=input_layer, outputs=output)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# 7) TRAIN THE MODEL
model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=5,        # You can increase to 10+ if you have enough data
    batch_size=64
)

# 8) EVALUATE
pred_probs = model.predict(X_val_pad)
y_pred = np.argmax(pred_probs, axis=1)

print("Classification Report:")
print(classification_report(y_val, y_pred))
