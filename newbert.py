import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset

# --- 1. Load and Prepare Data ---
try:
    # Try reading with default 'utf-8' encoding first
    df = pd.read_csv('data/1kreviews.csv')
except UnicodeDecodeError:
    # If 'utf-8' fails, try 'latin-1' or other relevant encodings
    try:
        df = pd.read_csv('data/1kreviews.csv', encoding='latin-1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        print("Please ensure 'data/1mreviews.csv' exists and is accessible.")
        exit() # Exit if the file cannot be loaded

# Ensure 'text' and 'stars' columns exist
if 'text' not in df.columns or 'stars' not in df.columns:
    print("Error: CSV must contain 'text' and 'stars' columns.")
    exit()

# Handle potential missing values in 'text'
df.dropna(subset=['text'], inplace=True)
# Convert text to string just in case
df['text'] = df['text'].astype(str)


def map_sentiment(star):
    if star in [1, 2]:
        return 0  # Negative
    elif star == 3:
        return 1  # Neutral
    else: # 4, 5 stars
        return 2  # Positive

df['label'] = df['stars'].apply(map_sentiment)

# Optional: Sample the data for faster testing/development
# df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# --- 2. Initialize Tokenizer and Model ---
MODEL_NAME = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# --- 3. Tokenization Function ---
def tokenize_function(examples):
    # Handle potential non-string data explicitly
    texts = [str(text) for text in examples['text']]
    return tokenizer(texts, padding="max_length", truncation=True, max_length=128) # Adjust max_length if needed

# Convert pandas DataFrame to Hugging Face Dataset
hg_dataset = Dataset.from_pandas(df[['text', 'label']])

# Tokenize the entire dataset
tokenized_dataset = hg_dataset.map(tokenize_function, batched=True)

# Remove the original text column as it's no longer needed by the model
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
# Rename 'label' to 'labels' which is expected by the Trainer
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
# Set format for PyTorch
tokenized_dataset.set_format("torch")


# --- 4. Stratified K-Fold Cross-Validation ---
N_SPLITS = 5
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
output_dir_base = 'output/distilbert'
os.makedirs(output_dir_base, exist_ok=True)

metrics_output_dir = os.path.join(output_dir_base, 'metrics')
os.makedirs(metrics_output_dir, exist_ok=True)


fold_metrics = []
all_preds = []
all_labels = []
all_probs = []

# Get labels for stratification
labels_for_stratification = df['label'].to_numpy()

for fold, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(tokenized_dataset)), labels_for_stratification)):
    print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")

    # --- Create Fold-Specific Datasets ---
    train_dataset = tokenized_dataset.select(train_idx)
    val_dataset = tokenized_dataset.select(val_idx)
    # Ensure validation set also has 'labels' column formatted correctly
    # (This step might be redundant if tokenized_dataset format is set correctly, but good for safety)
    val_dataset.set_format("torch", columns=['input_ids', 'attention_mask', 'labels'])

    # --- Define Model for the Fold ---
    # Load a fresh model for each fold to prevent knowledge leak
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # --- Training Arguments for the Fold ---
    fold_output_dir = os.path.join(output_dir_base, f'fold_{fold+1}')
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=1, # Keep low for DistilBERT fine-tuning, adjust as needed
        per_device_train_batch_size=16, # Adjust based on GPU memory
        per_device_eval_batch_size=32,  # Adjust based on GPU memory
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(fold_output_dir, 'logs'),
        logging_steps=50,
        evaluation_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch",       # Save model at the end of each epoch
        load_best_model_at_end=True, # Load the best model found during training
        metric_for_best_model="accuracy", # Use accuracy to determine the best model
        greater_is_better=True,
        report_to="none" # Disable default reporting (like wandb) unless configured
    )

    # --- Compute Metrics Function ---
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, average='weighted', zero_division=0)
        rec = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        return {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
        }

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, # Pass the correctly formatted validation dataset
        compute_metrics=compute_metrics,
        tokenizer=tokenizer, # Pass tokenizer for padding assistance if needed
    )

    # --- Train ---
    trainer.train()

    # --- Evaluate on Validation Set for the Fold ---
    print(f"Evaluating fold {fold+1}...")
    # Ensure val_dataset is used here if compute_metrics wasn't called during training's eval loop
    # or if you want final fold validation metrics explicitly
    results = trainer.predict(val_dataset)
    preds = np.argmax(results.predictions, axis=-1)
    probs = torch.softmax(torch.from_numpy(results.predictions), dim=-1).numpy()
    labels = results.label_ids # Get the true labels from the predict output

    # Ensure labels are not None and lengths match
    if labels is None or len(preds) != len(labels):
         print(f"Warning: Mismatch in prediction ({len(preds)}) and label ({len(labels) if labels is not None else 'None'}) lengths for fold {fold+1}. Skipping metric calculation for this fold.")
         # Handle this case - maybe load labels directly from val_idx if needed
         # current_fold_labels = df['label'].iloc[val_idx].to_numpy()
         # if len(preds) == len(current_fold_labels):
         #    labels = current_fold_labels
         # else:
         #    continue # Skip fold if label recovery fails
         continue # Skip fold

    fold_acc = accuracy_score(labels, preds)
    fold_prec = precision_score(labels, preds, average='weighted', zero_division=0)
    fold_rec = recall_score(labels, preds, average='weighted', zero_division=0)
    fold_f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    fold_cm = confusion_matrix(labels, preds)
    try:
        labels_binarized = label_binarize(labels, classes=[0, 1, 2])
        if labels_binarized.shape[1] == 1: # Handle case where only one class is present in fold validation
             fold_auc = 0.0 # Or handle as appropriate, AUC isn't well-defined
             print(f"Warning: Only one class present in validation set for fold {fold+1}. AUC set to 0.")
        else:
             fold_auc = roc_auc_score(labels_binarized, probs, multi_class='ovr', average='weighted')

    except ValueError as e:
        print(f"Warning: Could not calculate AUC for fold {fold+1}. Error: {e}")
        fold_auc = 0.0 # Assign a default value


    print(f"Fold {fold+1} Metrics:")
    print(f"  Accuracy:  {fold_acc:.4f}")
    print(f"  Precision: {fold_prec:.4f}")
    print(f"  Recall:    {fold_rec:.4f}")
    print(f"  F1 Score:  {fold_f1:.4f}")
    print(f"  AUC:       {fold_auc:.4f}")
    print(f"  Confusion Matrix:\n{fold_cm}")

    fold_metrics.append({
        'Fold': fold+1,
        'Accuracy': fold_acc,
        'Precision': fold_prec,
        'Recall': fold_rec,
        'F1 Score': fold_f1,
        'AUC': fold_auc
    })

    # Store predictions and labels for overall evaluation later
    all_preds.extend(preds.tolist())
    all_labels.extend(labels.tolist())
    all_probs.extend(probs.tolist())

    # Clean up GPU memory
    del model
    del trainer
    torch.cuda.empty_cache()


# --- 5. Aggregate and Save CV Metrics ---
cv_metrics_df = pd.DataFrame(fold_metrics)
cv_metrics_summary = cv_metrics_df.agg({
    'Accuracy': ['mean', 'std'],
    'Precision': ['mean', 'std'],
    'Recall': ['mean', 'std'],
    'F1 Score': ['mean', 'std'],
    'AUC': ['mean', 'std']
})

print("\n--- Cross-Validation Summary ---")
print(cv_metrics_summary)

# Save fold metrics and summary
cv_metrics_df.to_csv(os.path.join(metrics_output_dir, 'distilbert_cv_fold_metrics.csv'), index=False)
cv_metrics_summary.to_csv(os.path.join(metrics_output_dir, 'distilbert_cv_summary_metrics.csv'))


# --- 6. Final Evaluation (using aggregated CV predictions) ---
# This uses the predictions gathered from each fold's validation set
print("\n--- Overall Evaluation (Based on Aggregated CV Folds) ---")
overall_acc = accuracy_score(all_labels, all_preds)
overall_prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
overall_rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
overall_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
overall_cm = confusion_matrix(all_labels, all_preds)
all_labels_binarized = label_binarize(all_labels, classes=[0, 1, 2])
overall_auc = roc_auc_score(all_labels_binarized, all_probs, multi_class='ovr', average='weighted')


print(f"Overall Accuracy:  {overall_acc:.4f}")
print(f"Overall Precision: {overall_prec:.4f}")
print(f"Overall Recall:    {overall_rec:.4f}")
print(f"Overall F1 Score:  {overall_f1:.4f}")
print(f"Overall AUC:       {overall_auc:.4f}")

# Save overall metrics
overall_metrics_dict = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'CV Mean Accuracy', 'CV Std Accuracy'],
    'Score': [
        overall_acc, overall_prec, overall_rec, overall_f1, overall_auc,
        cv_metrics_summary.loc['mean', 'Accuracy'], cv_metrics_summary.loc['std', 'Accuracy']
    ]
}
overall_metrics_df = pd.DataFrame(overall_metrics_dict)
overall_metrics_df.to_csv(os.path.join(metrics_output_dir, 'distilbert_overall_metrics.csv'), index=False)


# --- 7. Plotting ---
plot_output_dir = os.path.join(output_dir_base, 'plots')
os.makedirs(plot_output_dir, exist_ok=True)

# Confusion Matrix
plt.figure(figsize=(7, 5))
sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('DistilBERT Overall Confusion Matrix (Aggregated CV)')
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, 'distilbert_overall_confusion_matrix.png'))
plt.close()
print(f"Overall Confusion Matrix saved to {plot_output_dir}")


# ROC Curve
plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i in range(3):
    fpr, tpr, _ = roc_curve(all_labels_binarized[:, i], np.array(all_probs)[:, i])
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc_score(all_labels_binarized[:, i], np.array(all_probs)[:, i]):.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DistilBERT Overall ROC Curve (Aggregated CV)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(plot_output_dir, 'distilbert_overall_roc_curve.png'))
plt.close()
print(f"Overall ROC Curve saved to {plot_output_dir}")

print("\nDistilBERT analysis complete.")