import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure you have these installed:
# !pip install transformers torch

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer

# ---------------
# 1) Data Loading
# ---------------
df = pd.read_csv("data/100kreviews.csv")  # Or your own CSV with more than 20k rows

# Map the star rating to numeric sentiment: 1/2 -> 0 (negative), 3 -> 1 (neutral), 4/5 -> 2 (positive)
def map_sentiment(star):
    if star in [1, 2]:
        return 0  # negative
    elif star == 3:
        return 1  # neutral
    else:
        return 2  # positive

df["label"] = df["stars"].apply(map_sentiment)

texts = df["text"].astype(str).tolist()
labels = df["label"].values

# 70-30 train/test split
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, stratify=labels, random_state=42
)

# ---------------
# 2) Tokenization
# ---------------
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# We'll define a simple PyTorch Dataset so we can feed it to the Trainer
class YelpDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze() for key, val in encodings.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

# ---------------
# 3) 5-Fold Cross Validation on the training set
#    We'll do a quick loop that trains DistilBERT for a small number of epochs.
#    *If your dataset is very large, consider fewer folds or fewer epochs to save time.
# ---------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []

# Hyperparams for cross validation (shorter/smaller for speed)
cv_training_args = TrainingArguments(
    output_dir="output/distilbert_cv",     # Folder to store intermediate files
    num_train_epochs=1,                    # Fewer epochs for speed in CV
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="no",                    # Don't save to disk each epoch
    disable_tqdm=True                      # Quieter training
)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_texts, y_train)):
    # Split the training set into fold-train and fold-validation
    fold_train_texts = [X_train_texts[i] for i in train_idx]
    fold_train_labels = y_train[train_idx]
    fold_val_texts = [X_train_texts[i] for i in val_idx]
    fold_val_labels = y_train[val_idx]

    train_dataset = YelpDataset(fold_train_texts, fold_train_labels, tokenizer)
    val_dataset = YelpDataset(fold_val_texts, fold_val_labels, tokenizer)

    # Load a fresh DistilBERT model each time
    model_cv = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3  # negative, neutral, positive
    )

    def compute_metrics(eval_pred):
        logits, labels_cv = eval_pred
        preds_cv = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels_cv, preds_cv)
        return {"accuracy": acc}

    trainer_cv = Trainer(
        model=model_cv,
        args=cv_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer_cv.train()
    # Evaluate on the fold's validation set
    eval_results = trainer_cv.evaluate()
    fold_acc = eval_results["eval_accuracy"]
    cv_accuracies.append(fold_acc)
    print(f"Fold {fold_idx+1} - Accuracy: {fold_acc:.4f}")

print("Cross Validation Accuracies:", cv_accuracies)
print("Mean CV Accuracy:", np.mean(cv_accuracies))

# ---------------
# 4) Final Training on ALL training data, then evaluation on the 30% test set
# ---------------
train_dataset_full = YelpDataset(X_train_texts, y_train, tokenizer)
test_dataset = YelpDataset(X_test_texts, y_test, tokenizer)

# Feel free to increase epochs if you have time/resources
training_args = TrainingArguments(
    output_dir="output/distilbert_final",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=200,
    evaluation_strategy="epoch",
    save_strategy="no",
    disable_tqdm=False
)

model_final = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

# We'll define a function to compute the main metrics: 
# (Accuracy, Precision, Recall, F1) for the Trainer
def compute_metrics_full(eval_pred):
    logits, labels_ = eval_pred
    preds_ = np.argmax(logits, axis=-1)
    acc_ = accuracy_score(labels_, preds_)
    prec_ = precision_score(labels_, preds_, average="weighted")
    rec_ = recall_score(labels_, preds_, average="weighted")
    f1_ = f1_score(labels_, preds_, average="weighted")
    return {
        "accuracy": acc_,
        "precision": prec_,
        "recall": rec_,
        "f1": f1_
    }

trainer = Trainer(
    model=model_final,
    args=training_args,
    train_dataset=train_dataset_full,
    eval_dataset=test_dataset,         # We'll check test performance each epoch
    compute_metrics=compute_metrics_full
)

trainer.train()

# ---------------
# 5) Final Evaluation + Additional Metrics (Confusion Matrix, ROC, AUC)
# ---------------
pred_output = trainer.predict(test_dataset)
test_logits = pred_output.predictions
test_labels = pred_output.label_ids
test_preds = np.argmax(test_logits, axis=-1)

acc = accuracy_score(test_labels, test_preds)
prec = precision_score(test_labels, test_preds, average="weighted")
rec = recall_score(test_labels, test_preds, average="weighted")
f1 = f1_score(test_labels, test_preds, average="weighted")

# Confusion Matrix
cm = confusion_matrix(test_labels, test_preds)

# For multi-class AUC we need probabilities, plus one-vs-rest
test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=-1).numpy()
# We'll binarize y_test for multi-class AUC
y_test_binarized = np.zeros((len(test_labels), 3))
for i, lbl in enumerate(test_labels):
    y_test_binarized[i, lbl] = 1

# One-vs-Rest AUC
auc = roc_auc_score(y_test_binarized, test_probs, multi_class='ovr')

# ---------------
# 6) Save Plots & CSV
# ---------------
os.makedirs("output/matrix", exist_ok=True)

# Confusion Matrix Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Neg", "Neu", "Pos"],
            yticklabels=["Neg", "Neu", "Pos"])
plt.title("DistilBERT - Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output/matrix/distilbert_confusion_matrix.png")
plt.close()

# ROC curve (OvR). We'll plot 3 separate ROC curves (one per class).
plt.figure()
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], test_probs[:, i])
    plt.plot(fpr, tpr, label=f"Class {i}")
plt.title("DistilBERT - ROC Curve (Test Set)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("output/matrix/distilbert_roc_curve.png")
plt.close()

# Print final metrics
print("=== DistilBERT Final Test Performance ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")
print(f"Mean CV Accuracy (5-Fold): {np.mean(cv_accuracies):.4f}")

# Save metrics to CSV
metrics_df = pd.DataFrame({
    "Accuracy": [acc],
    "Precision": [prec],
    "Recall": [rec],
    "F1 Score": [f1],
    "AUC": [auc],
    "CV Mean Accuracy": [np.mean(cv_accuracies)]
})
metrics_df.to_csv("output/matrix/distilbert_metrics.csv", index=False)

print("\nSaved confusion matrix, ROC curve, and metrics in 'output/matrix' folder.")