import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

from datasets import Dataset as HFDataset, DatasetDict
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -----------------------------
# 1. DEVICE SETUP
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 2. LOAD DATA
# -----------------------------
file_path = "data/1kreviews.csv"  # example path; update to your actual CSV
df = pd.read_csv(file_path)

# Clean up column names if needed
df.columns = df.columns.str.strip()

# Check columns
print("Column Names in CSV:", df.columns.tolist())

if 'stars' not in df.columns or 'text' not in df.columns:
    raise KeyError("Please ensure your CSV has columns named 'stars' and 'text'.")

# Function to safely parse text if needed
def safe_parse(text):
    try:
        return ' '.join(ast.literal_eval(text)) if isinstance(text, str) else text
    except (SyntaxError, ValueError):
        return text

df['text'] = df['text'].apply(safe_parse)

# -----------------------------
# 3. LABEL ENCODING (3-CLASS)
# -----------------------------
# 1 or 2 stars  -> 0 (negative)
# 3 star       -> 1 (neutral)
# 4 or 5 stars -> 2 (positive)
def encode_sentiment(stars):
    if stars <= 2:
        return 0  # negative
    elif stars == 3:
        return 1  # neutral
    else:
        return 2  # positive

df['label'] = df['stars'].apply(encode_sentiment)

# -----------------------------
# 4. TRAIN-TEST SPLIT (70/30)
# -----------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df['label'],
    random_state=42
)

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train size: {train_df.shape[0]}; Test size: {test_df.shape[0]}")

# -----------------------------
# 5. TOKENIZATION
# -----------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# -----------------------------
# Convert to Hugging Face Dataset
# -----------------------------
train_dataset_hf = HFDataset.from_pandas(train_df[['text', 'label']])
test_dataset_hf  = HFDataset.from_pandas(test_df[['text', 'label']])

# We apply the tokenizer using Dataset.map
train_dataset_hf = train_dataset_hf.map(tokenize, batched=True)
test_dataset_hf  = test_dataset_hf.map(tokenize, batched=True)

# Set format to pytorch
train_dataset_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# -----------------------------
# 6. CROSS-VALIDATION PREP
# -----------------------------
# We'll do 5-fold cross-validation on train_df
# and keep track of metrics across folds.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = {
    "accuracy": [],
    "precision_macro": [],
    "recall_macro": [],
    "f1_macro": []
}

# Convert the training set to a DataFrame again for manual folds
X_train_text = train_df["text"].values
y_train = train_df["label"].values

# For quickly building HF datasets in each fold
def build_hf_dataset(texts, labels):
    tmp_df = pd.DataFrame({"text": texts, "label": labels})
    dset = HFDataset.from_pandas(tmp_df)
    dset = dset.map(tokenize, batched=True)
    dset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return dset

# -----------------------------
# 7. CROSS-VALIDATION LOOP
# -----------------------------
print("\n--- 5-Fold Cross Validation ---")

fold_idx = 1
for train_index, val_index in skf.split(X_train_text, y_train):
    print(f"\nFold {fold_idx}:")

    fold_idx += 1

    # Split into fold train/val
    X_fold_train, X_fold_val = X_train_text[train_index], X_train_text[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

    hf_train = build_hf_dataset(X_fold_train, y_fold_train)
    hf_val   = build_hf_dataset(X_fold_val,   y_fold_val)

    # Create a fresh DistilBERT model each fold
    model_fold = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3  # 3-class
    ).to(device)

    training_args_fold = TrainingArguments(
        output_dir="./results_fold",  # each fold's output (overwritten each time)
        evaluation_strategy="epoch",
        save_strategy="no",   # For quick CV, you can skip model checkpointing 
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,    # lowered for example speed
        weight_decay=0.01
    )

    trainer_fold = Trainer(
        model=model_fold,
        args=training_args_fold,
        train_dataset=hf_train,
        eval_dataset=hf_val
    )

    # Train on the fold
    trainer_fold.train()

    # Evaluate on the fold's validation
    preds = trainer_fold.predict(hf_val)
    y_pred = np.argmax(preds.predictions, axis=1)
    y_true = y_fold_val

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='macro', zero_division=0)

    fold_metrics["accuracy"].append(acc)
    fold_metrics["precision_macro"].append(prec)
    fold_metrics["recall_macro"].append(rec)
    fold_metrics["f1_macro"].append(f1)

    print(f"Fold metrics - Acc: {acc:.4f}, Prec(macro): {prec:.4f}, "
          f"Rec(macro): {rec:.4f}, F1(macro): {f1:.4f}")

# -----------------------------
# 8. AVERAGE CROSS-VAL RESULTS
# -----------------------------
print("\n=== Cross-Validation Summary (5-fold) ===")
mean_acc  = np.mean(fold_metrics["accuracy"])
mean_prec = np.mean(fold_metrics["precision_macro"])
mean_rec  = np.mean(fold_metrics["recall_macro"])
mean_f1   = np.mean(fold_metrics["f1_macro"])

print(f"Mean Accuracy:      {mean_acc:.4f}")
print(f"Mean Precision(m):  {mean_prec:.4f}")
print(f"Mean Recall(m):     {mean_rec:.4f}")
print(f"Mean F1(m):         {mean_f1:.4f}")

# -----------------------------
# 9. FINAL TRAIN (on full 70%) + TEST
# -----------------------------
# After cross-validation, you typically pick best hyperparams or confirm you want default.
# Retrain on the entire 70% training set for final evaluation on 30% test.

final_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3
).to(device)

final_training_args = TrainingArguments(
    output_dir="./results_final",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # can increase after testing
    weight_decay=0.01,
    load_best_model_at_end=False
)

final_trainer = Trainer(
    model=final_model,
    args=final_training_args,
    train_dataset=train_dataset_hf,
    eval_dataset=test_dataset_hf
)

# Final training on full 70%
final_trainer.train()

# -----------------------------
# 10. EVALUATE ON 30% TEST
# -----------------------------
pred_test = final_trainer.predict(test_dataset_hf)
y_pred_test = np.argmax(pred_test.predictions, axis=1)
y_true_test = test_df["label"].tolist()

accuracy = accuracy_score(y_true_test, y_pred_test)
precision = precision_score(y_true_test, y_pred_test, average="macro", zero_division=0)
recall = recall_score(y_true_test, y_pred_test, average="macro", zero_division=0)
f1 = f1_score(y_true_test, y_pred_test, average="macro", zero_division=0)

print("\n=== Final Test Set Performance ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}\n")

# Classification report
class_rep = classification_report(y_true_test, y_pred_test, zero_division=0)
print("Classification Report:\n", class_rep)

# -----------------------------
# 11. SAVE OUTPUTS
# -----------------------------
# Create output folder if not exists
os.makedirs("output/matrix", exist_ok=True)

# Confusion Matrix Plot
cm = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative(0)', 'Neutral(1)', 'Positive(2)'],
            yticklabels=['Negative(0)', 'Neutral(1)', 'Positive(2)'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("output/matrix/confusion_matrix.png")  # Save figure
plt.close()

# Save classification report to text file
with open("output/matrix/classification_report.txt", "w") as f:
    f.write(class_rep)

# -----------------------------
# 12. ROC & AUC
# For multi-class, we do One-vs-Rest approach
# -----------------------------
# One-vs-Rest binarization
# shape = (#samples, #classes)
y_true_binarized = np.zeros((len(y_true_test), 3))
for i, label_id in enumerate(y_true_test):
    y_true_binarized[i, label_id] = 1

# Softmax on predictions to get probabilities
probs = torch.softmax(torch.tensor(pred_test.predictions), dim=1).numpy()

# Calculate macro-average AUC
auc_score = roc_auc_score(y_true_binarized, probs, average="macro", multi_class="ovr")
print(f"Multiclass Macro-AUC: {auc_score:.4f}")

# For plotting a multiclass ROC curve, we can plot each class vs rest
fpr = {}
tpr = {}
roc_auc = {}

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], probs[:, i])
    roc_auc[i] = roc_auc_score(y_true_binarized[:, i], probs[:, i])

# Plot the ROC curves for each class
plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (area = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (Test Set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("output/matrix/roc_curves.png")
plt.close()

print("All outputs saved to 'output/matrix' folder.")
