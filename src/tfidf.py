import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from scipy.sparse import save_npz

file_path = "data/lemmatized_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8")

#convert string representation of lists into actual text
def safe_convert(text):
    try:
        return " ".join(ast.literal_eval(text))
    except (SyntaxError, ValueError):
        return str(text)

df["lemmatized_text"] = df["lemmatized_text"].apply(safe_convert)

#filter only English text
def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

df = df[df["lemmatized_text"].apply(is_english)]  #only english reviews

#removes numbers
vectorizer = TfidfVectorizer(token_pattern=r"\b[a-zA-Z]{2,}\b", stop_words="english")

#transform the text data (Sparse Matrix)
tfidf_matrix = vectorizer.fit_transform(df["lemmatized_text"])

save_npz("tfidf_sparse_matrix.npz", tfidf_matrix)
print("Sparse TF-IDF matrix saved as 'tfidf_sparse_matrix.npz'.")

#save feature names (vocabulary)
np.save("tfidf_feature_names.npy", vectorizer.get_feature_names_out())
print("TF-IDF vocabulary saved as 'tfidf_feature_names.npy'.")

#convert to a DataFrame for better visualization
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

print("TF-IDF Vectorized Data (Top 5 Rows):")
print(tfidf_df.head())

#top 20 words with highest TF-IDF scores
first_doc_tfidf = tfidf_df.iloc[0].sort_values(ascending=False)
print("\nTop 20 words in first document by TF-IDF score:")
print(first_doc_tfidf.head(20))
