import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
import pickle

# Load dataset
file_path = "data/lemmatized_dataset.csv"
df = pd.read_csv(file_path)

# Ensure the column exists
if 'lemmatized_text' not in df.columns:
    raise ValueError("Column 'lemmatized_text' not found in the dataset")

# Convert lemmatized_text column to list of tokenized words (assuming it is space-separated)
tokenized_texts = df['lemmatized_text'].apply(lambda x: x.split()).tolist()

# Train Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_texts, vector_size=100, window=5, min_count=2, workers=4)

# Function to get document vector by averaging word vectors
def get_document_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)  # Return zero vector if no words are found in Word2Vec model
    return np.mean(vectors, axis=0)

# Convert all documents into vectors
document_vectors = np.array([get_document_vector(tokens, word2vec_model) for tokens in tokenized_texts])

# Convert to sparse matrix
sparse_matrix = csr_matrix(document_vectors)

# Save sparse matrix
sparse_matrix_path = "data/sparse_matrix.pkl"
with open(sparse_matrix_path, 'wb') as f:
    pickle.dump(sparse_matrix, f)

print(f"Sparse matrix saved at {sparse_matrix_path}")
