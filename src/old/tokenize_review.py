import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

df = pd.read_csv("cleaned_reviews.csv")


#tokenize
df["tokenized_review"] = df["text"].apply(lambda x: word_tokenize(str(x)))

df.to_csv("tokenized_reviews.csv", index=False)
 