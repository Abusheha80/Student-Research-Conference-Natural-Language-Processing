# yelpReview

This project processes Yelp reviews through various stages to clean, tokenize, handle negation, lemmatize, and vectorize the text data. Follow the steps below to run the project:

1. **Convert JSON to CSV**
   - Run `json_to_csv.py` to convert the Yelp JSON dataset to a CSV file.
   - This will create `sorted_reviews.csv`.

2. **Clean the Data**
   - Run `clean.py` to remove punctuation and other unwanted characters.
   - This will save the cleaned data to `cleaned_reviews.csv`.

3. **Tokenize the Reviews**
   - Run `tokenize_review.py` to tokenize the reviews.
   - This will generate `tokenized_reviews.csv`.

4. **Handle Negation**
   - Run `negation.py` to add a column that handles negation in the reviews.
   - This will produce `negation_reviews.csv`.

5. **Lemmatize the Text**
   - Run `lemmatization.py` to lemmatize the text and add a new column `lemmatized_text`.
   - This will save the lemmatized data to `lemmatized_reviews.csv`.

6. **Create a Separate Dataset**
   - Run `after_lemmatized.py` to convert the lemmatized data into a separate dataset.
   - This will create `lemmatized_dataset.csv`.

7. **Vectorize the Data**
   - Run `vectorize.py` or `tfidf.py` to vectorize the new dataset using TF-IDF.
   - This will generate a TF-IDF vectorized dataset for further analysis.

## Running the Scripts

To run each script, use the following command in your terminal:

```sh
python script_name.py