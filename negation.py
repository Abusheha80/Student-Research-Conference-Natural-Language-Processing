import pandas as pd
import ast

#negation will stop
clause_breakers = {"but", "because", "however", "although", "though", "yet"}

# negation will not apply
negative_words = {"bad", "worse", "worst", "terrible", "horrible", "awful", "disgusting"}

def handle_negation(tokenized_text, negation_scope=3):
    negation_words = {"not", "no", "never", "n't"}
    processed_tokens = []
    negate_count = 0 

    for token in tokenized_text:
        if token in negation_words:
            negate_count = negation_scope  # Set negation for the next few words
            processed_tokens.append(token)
        elif negate_count > 0:
            if token in clause_breakers or token in negative_words:
                negate_count = 0
            else:
                processed_tokens.append(f"not_{token}")  
                negate_count -= 1 
        else:
            processed_tokens.append(token)  

    return processed_tokens

file_path = "tokenized_reviews.csv"
df = pd.read_csv(file_path)

df["tokenized_review"] = df["tokenized_review"].apply(ast.literal_eval)

df["negation_review"] = df["tokenized_review"].apply(handle_negation)

df.to_csv("negation_reviews.csv", index=False)