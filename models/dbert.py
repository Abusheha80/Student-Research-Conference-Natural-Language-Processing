import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)

df = pd.read_csv("data/100kreviews.csv")

def map_sentiment(star):
    return 0 if star in [1, 2] else 1 if star == 3 else 2

df["label"] = df["stars"].apply(map_sentiment)

texts = df["text"].astype(str).tolist()
labels = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, stratify=labels, random_state=42)

#tokenization
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

class YelpDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = YelpDataset(X_train, y_train, tokenizer)
test_dataset = YelpDataset(X_test, y_test, tokenizer)

training_args = TrainingArguments(
    output_dir="output/distilbert_final",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
    gradient_accumulation_steps=2,
    logging_steps=100,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)

trainer = Trainer(
    model=model_final,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

#eval
pred_output = trainer.predict(test_dataset)
test_logits = pred_output.predictions
test_labels = pred_output.label_ids
test_preds = np.argmax(test_logits, axis=-1)

#metrics
acc = accuracy_score(test_labels, test_preds)
prec, rec, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="weighted")
cm = confusion_matrix(test_labels, test_preds)
probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
auc = roc_auc_score(pd.get_dummies(test_labels), probs, multi_class='ovr')

#save results
os.makedirs("output/matrix", exist_ok=True)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Neg','Neu','Pos'], yticklabels=['Neg','Neu','Pos'])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("output/matrix/distilbert_confusion_matrix.png")

#plot roc
plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve(pd.get_dummies(test_labels).values[:, i], probs[:, i])
    plt.plot(fpr, tpr, label=f'Class {i}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig("output/matrix/distilbert_roc_curve.png")

metrics_df = pd.DataFrame({"Accuracy": [acc], "Precision": [prec], "Recall": [rec], "F1 Score": [f1], "AUC": [auc]})
metrics_df.to_csv("output/matrix/distilbert_metrics.csv", index=False)

print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
