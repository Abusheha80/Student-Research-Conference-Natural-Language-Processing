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

# 4. Calculate and Print Sentiment Percentages
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
print(sentiment_counts)