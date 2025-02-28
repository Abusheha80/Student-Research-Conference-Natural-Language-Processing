import pandas as pd
import spacy
from tqdm import tqdm
import multiprocessing as mp

# Load only components we need (disable unnecessary pipelines)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Set up batch processing parameters
batch_size = 1000  # Adjust based on your memory

def lemmatize_batch(texts):
    # Process multiple texts as a batch
    docs = list(nlp.pipe(texts))
    return [" ".join([token.lemma_ for token in doc]) for doc in docs]

def process_in_batches(df, text_column):
    lemmatized_texts = []
    texts = df[text_column].astype(str).tolist()
    
    # Calculate number of batches
    num_batches = (len(texts) // batch_size) + (1 if len(texts) % batch_size != 0 else 0)
    
    # Process in batches with progress bar
    for i in tqdm(range(num_batches), desc="Lemmatizing"):
        batch_texts = texts[i * batch_size:(i + 1) * batch_size]
        lemmatized_batch = lemmatize_batch(batch_texts)
        lemmatized_texts.extend(lemmatized_batch)
    
    return lemmatized_texts

if __name__ == "__main__":
    # Set number of threads for spaCy
    spacy.require_cpu()
    num_cpus = mp.cpu_count()
    spacy.prefer_gpu()  # Will use GPU if available
    
    print(f"Loading data from 'negation_reviews.csv'...")
    file_path = "negation_reviews.csv"
    df = pd.read_csv(file_path)
    
    print(f"Lemmatizing {len(df)} reviews using {num_cpus} CPU cores...")
    df["lemmatized_text"] = process_in_batches(df, "negation_review")
    
    output_path = "lemmatized_reviews.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Lemmatized dataset saved to {output_path}")
