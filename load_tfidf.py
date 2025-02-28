import pandas as pd
from scipy.sparse import load_npz
import numpy as np

tfidf_matrix = load_npz("tfidf_sparse_matrix.npz")
print("Sparse TF-IDF matrix loaded successfully.")

#load the vocabulary
feature_names = np.load("tfidf_feature_names.npy", allow_pickle=True)
print("TF-IDF vocabulary loaded successfully.")

# sparse matrix to a df
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

tfidf_df.to_csv("tfidf_matrix_loaded.csv", index=False)
print("TF-IDF matrix saved as 'tfidf_matrix_loaded.csv'.")
