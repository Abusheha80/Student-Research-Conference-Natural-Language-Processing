import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

df = pd.read_csv("data/10kreviews.csv")
df = df.rename(columns={'stars': 'label'})

df['label'] = df['label'].apply(lambda x: 1 if x >= 3 else 0)

df['label'] = df['label'].astype(int)

print("Columns in the dataset:", df.columns.tolist())
print(df.head())

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',              
    num_train_epochs=3,                  
    per_device_train_batch_size=16,      
    per_device_eval_batch_size=64,       
    evaluation_strategy="steps",         
    eval_steps=500,                      
    save_steps=500,                      
    warmup_steps=500,                    
    weight_decay=0.01,                   
    logging_dir='./logs',                
    logging_steps=100,                   
    load_best_model_at_end=True,         
    metric_for_best_model="accuracy",    
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

model.save_pretrained("data/fine-tuned-bert-sentiment")
tokenizer.save_pretrained("data/fine-tuned-bert-sentiment")
