import pandas as pd
import ast
from langdetect import detect

df = pd.read_csv("lemmatized_reviews.csv")

# 1) Convert the stringified token lists into normal text
def safe_convert(text):
    try:
        return " ".join(ast.literal_eval(text))
    except (SyntaxError, ValueError):
        return str(text)

df["lemmatized_text"] = df["lemmatized_text"].apply(safe_convert)

def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False

df = df[df["lemmatized_text"].apply(is_english)]

df.drop([
    "review_id", 
    "user_id", 
    "useful", 
    "funny", 
    "cool", 
    "text",            
    "tokenized_review",
    "negation_review"
], axis=1, inplace=True)

df.rename(columns={"lemmatized_text": "text"}, inplace=True)

df.to_csv("reviews.csv", index=False)
print("Saved filtered dataset to reviews.csv with shape:", df.shape)
