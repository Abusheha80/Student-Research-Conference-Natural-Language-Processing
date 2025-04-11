import json
import heapq
import re
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import spacy
from langdetect import detect

# ------------------ Setup ------------------
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(nltk.corpus.stopwords.words('english'))

# ------------------ Step 1: Load JSON and Select Reviews ------------------
def parse_date(review):
    return datetime.strptime(review['date'], '%Y-%m-%d %H:%M:%S')

def load_and_select_reviews(json_path, top_n=100000):
    reviews = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line.strip())
                reviews.append(review)
            except json.JSONDecodeError:
                continue
    top_reviews = heapq.nlargest(top_n, reviews, key=parse_date)
    return pd.DataFrame(top_reviews)

# ------------------ Step 2: Cleaning ------------------
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "n't": " not",
            # Add more contractions if needed
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        words = [word for word in words if not re.search(r'(.)\1{2,}', word)]
        words = [word for word in words if word not in stop_words]
        return ' '.join(words).strip()
    return text

def tokenize_text(text):
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        return word_tokenize(str(text))
    except LookupError:
        # Fallback method if word_tokenize fails
        return text.split()

clause_breakers = {"but", "because", "however", "although", "though", "yet"}
negative_words = {"bad", "worse", "worst", "terrible", "horrible", "awful", "disgusting"}

def handle_negation(tokenized_text, negation_scope=3):
    negation_words = {"not", "no", "never", "n't"}
    processed_tokens = []
    negate_count = 0
    for token in tokenized_text:
        if token in negation_words:
            negate_count = negation_scope
            processed_tokens.append(token)
        elif negate_count > 0:
            if token in clause_breakers or token in negative_words:
                negate_count = 0
                processed_tokens.append(token)
            else:
                processed_tokens.append(f"not_{token}")
                negate_count -= 1
        else:
            processed_tokens.append(token)
    return processed_tokens

def lemmatize_texts(texts, batch_size=100000):
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    lemmatized_texts = []
    num_batches = (len(texts) // batch_size) + (1 if len(texts) % batch_size != 0 else 0)
    for i in tqdm(range(num_batches), desc="Lemmatizing"):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        docs = list(nlp.pipe(batch_texts))
        lemmatized_batch = [" ".join([token.lemma_ for token in doc]) for doc in docs]
        lemmatized_texts.extend(lemmatized_batch)
    return lemmatized_texts

def is_english(text):
    try:
        return detect(text) == "en"
    except Exception:
        return False

# ------------------ Main Pipeline ------------------
if __name__ == "__main__":

    json_path = "data/yelp_academic_dataset_review.json"
    print("Loading reviews from JSON...")
    df = load_and_select_reviews(json_path)

    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    print("Tokenizing text...")
    df['tokenized_review'] = df['cleaned_text'].apply(tokenize_text)

    print("Handling negation...")
    df['negation_review'] = df['tokenized_review'].apply(handle_negation)
    df['negation_review_str'] = df['negation_review'].apply(lambda tokens: " ".join(tokens))

    print("Lemmatizing text...")
    texts_to_lemmatize = df['negation_review_str'].tolist()
    df['lemmatized_text'] = lemmatize_texts(texts_to_lemmatize)

    df = df[df['lemmatized_text'].apply(is_english)]

    # Assign sentiment
    def assign_sentiment(star):
        if star >= 4:
            return 'positive'
        elif star == 3:
            return 'neutral'
        else:
            return 'negative'

    df['sentiment'] = df['stars'].apply(assign_sentiment)

    # Save unbalanced dataset
    unbalanced_csv_path = "data/100kreviews_unbalanced.csv"
    df[['business_id', 'stars', 'date', 'lemmatized_text', 'sentiment']].rename(columns={'lemmatized_text': 'text'}).to_csv(unbalanced_csv_path, index=False)
    print(f"Saved unbalanced dataset to {unbalanced_csv_path} with shape: {df.shape}")

    # Balance Classes by Undersampling
    print("Balancing sentiment distribution...")
    min_class_size = df['sentiment'].value_counts().min()

    balanced_df = pd.concat([
        df[df['sentiment'] == sentiment].sample(min_class_size, random_state=42)
        for sentiment in df['sentiment'].unique()
    ])

    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Balanced sentiment proportions:")
    print(balanced_df['sentiment'].value_counts(normalize=True) * 100)

    final_df = balanced_df[['business_id', 'stars', 'date', 'lemmatized_text', 'sentiment']].rename(columns={'lemmatized_text': 'text'})

    final_csv_path = "data/100kreviews.csv"
    final_df.to_csv(final_csv_path, index=False)
    print(f"Saved balanced dataset to {final_csv_path} with shape: {final_df.shape}")
