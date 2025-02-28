import pandas as pd

lemmatized_reviews = pd.read_csv("lemmatized_reviews.csv")

lemmatized_text_dataset = lemmatized_reviews[['lemmatized_text']]

#save
lemmatized_text_dataset.to_csv("lemmatized_dataset.csv", index=False)

print("New dataset saved as 'lemmatized_dataset.csv'")
