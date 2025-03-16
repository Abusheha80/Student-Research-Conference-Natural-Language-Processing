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

def load_and_select_reviews(json_path, top_n=90000):
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
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he'll've": "he will have",
            "he's": "he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'd've": "i would have",
            "i'll": "i will",
            "i'll've": "i will have",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'd've": "it would have",
            "it'll": "it will",
            "it'll've": "it will have",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she would",
            "she'd've": "she would have",
            "she'll": "she will",
            "she'll've": "she will have",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they would",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have"
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

# tokenization
def tokenize_text(text):
    return word_tokenize(str(text))

# negation handling
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

# lemmatization
def lemmatize_texts(texts, batch_size=1000):
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    lemmatized_texts = []
    num_batches = (len(texts) // batch_size) + (1 if len(texts) % batch_size != 0 else 0)
    for i in tqdm(range(num_batches), desc="Lemmatizing"):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        docs = list(nlp.pipe(batch_texts))
        lemmatized_batch = [" ".join([token.lemma_ for token in doc]) for doc in docs]
        lemmatized_texts.extend(lemmatized_batch)
    return lemmatized_texts

#accepting only english reviews
def is_english(text):
    try:
        return detect(text) == "en"
    except Exception:
        return False

# ------------------ Main Pipeline ------------------
if __name__ == "__main__":
    # Load JSON reviews and select the latest 10,000
    json_path = "data/yelp_academic_dataset_review.json"
    print("Loading reviews from JSON...")
    df = load_and_select_reviews(json_path)
    
    # Clean the original review text
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Tokenize the cleaned text
    print("Tokenizing text...")
    df['tokenized_review'] = df['cleaned_text'].apply(tokenize_text)
    
    # Handle negation
    print("Handling negation...")
    df['negation_review'] = df['tokenized_review'].apply(handle_negation)
    df['negation_review_str'] = df['negation_review'].apply(lambda tokens: " ".join(tokens))
    
    # Lemmatize the negation-processed text
    print("Lemmatizing text...")
    texts_to_lemmatize = df['negation_review_str'].tolist()
    df['lemmatized_text'] = lemmatize_texts(texts_to_lemmatize)
    
    # final processing

    df = df[df['lemmatized_text'].apply(is_english)]
    
    # dropping unnecessary columns
    final_df = df[['business_id', 'stars', 'date']].copy()
    final_df['text'] = df['lemmatized_text']
    
    # saving output
    final_csv_path = "data/reviews.csv"
    final_df.to_csv(final_csv_path, index=False)
    print(f"Saved filtered dataset to {final_csv_path} with shape: {final_df.shape}")
