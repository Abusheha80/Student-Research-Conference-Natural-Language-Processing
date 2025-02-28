import json
import csv
from datetime import datetime
import heapq

json_path = "yelp_academic_dataset_review.json"
csv_path = "sorted_reviews.csv"

#function to parse date from review
def parse_date(review):
    return datetime.strptime(review['date'], '%Y-%m-%d %H:%M:%S')

#read and process reviews incrementally
reviews = []
with open(json_path, 'r', encoding='utf-8') as json_file:
    for line in json_file:
        try:
            review = json.loads(line.strip())
            reviews.append(review)
        except json.JSONDecodeError:
            print(f"Skipping malformed JSON at line {len(reviews) + 1}")

#get the latest 10,000 reviews by date
top_reviews = heapq.nlargest(10000, reviews, key=parse_date)

#rrite to the CSV file
with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=top_reviews[0].keys())
    writer.writeheader()

    for review in top_reviews:
        writer.writerow(review)

print(f"CSV file created successfully with {min(10000, len(reviews))} records.")
