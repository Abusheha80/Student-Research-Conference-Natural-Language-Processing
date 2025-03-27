import pandas as pd

# 1. Read Data
df = pd.read_csv("data/10kreviews.csv")

# 2. Create Sentiment Labels from 'stars'
def label_sentiment(stars):
    if (stars >= 4):
        return 'positive'
    elif (stars == 3):
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['stars'].apply(label_sentiment)

# 3. Save the DataFrame with Sentiment Column
df.to_csv("data/10kreviews_with_sentiment.csv", index=False)

# 4. Calculate and Print Sentiment Percentages
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100 
print(sentiment_counts)

# X = df['text']
# y = df['sentiment']

# 4. Train-Test Split
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, 
#     y, 
#     test_size=0.2, 
#     random_state=42  
# )

# # 5. Build a Pipeline with TF-IDF + Logistic Regression
# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# model = Pipeline([
#     ('tfidf', TfidfVectorizer()),     # Convert text to TF-IDF vectors
#     ('clf', LogisticRegression())     # Classifier
# ])

# # 6. Train the Model
# model.fit(X_train, y_train)

# # 7. Evaluate the Model
# from sklearn.metrics import classification_report

# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# # 8. (Optional) Save the Model for Future Use
# import joblib
# joblib.dump(model, 'output/sentiment_model.pkl')
