import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    df = pd.read_csv("1mreviews.csv") 
    
    def map_stars_to_sentiment(star):
        if star in [1, 2]:
            return "negative"
        elif star == 3:
            return "neutral"
        else:
            return "positive"

    df['label'] = df['stars'].apply(map_stars_to_sentiment)

    X = df['text'].astype(str)  
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.30, 
        stratify=y,      # keeps label distribution consistent
        random_state=42
    )

    # ================================
    # 5. Create a Pipeline for SVM
    # ================================
    # We will use TF-IDF + SVM in one pipeline,
    # and then use GridSearchCV for parameter tuning.

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("svm", SVC(probability=True))  # probability=True to get ROC–AUC
    ])

    # ==============================
    # 6. Define Hyperparameter Grid
    # ==============================
    # Adjust or expand if needed. The below tries different
    # C values and kernel types. 'balanced' class_weight can help
    # if there's an imbalance in negative/neutral/positive distribution.
    param_grid = {
        "tfidf__min_df": [1, 2],
        "tfidf__max_df": [0.9, 1.0],
        "svm__C": [0.1, 1, 10],
        "svm__kernel": ["linear", "rbf"],
        "svm__class_weight": [None, 'balanced']
    }

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="accuracy",  # or use "f1_weighted" if you want an F1-based search
        cv=cv,
        n_jobs=-1,          # use all processors for speed
        verbose=2
    )

    # ================================
    # 7. Fit Model with GridSearchCV
    # ================================
    grid_search.fit(X_train, y_train)

    # Best pipeline from cross-validation
    best_model = grid_search.best_estimator_

    print("Best CV params:", grid_search.best_params_)
    print("Best CV score:", grid_search.best_score_)

    # ================================
    # 8. Evaluate on Test Set
    # ================================
    y_pred = best_model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "neutral", "positive"])

    # Because we have three classes, compute the AUC by treating the problem
    # with a One-vs-Rest approach (or micro/macro average).
    # We must binarize the labels for a multi-class AUC:
    from sklearn.preprocessing import label_binarize
    classes = ["negative", "neutral", "positive"]
    y_test_binarized = label_binarize(y_test, classes=classes)
    y_pred_prob      = best_model.predict_proba(X_test)  # shape: (n_samples, n_classes)

    # Macro-averaged AUC across the three classes
    auc = roc_auc_score(y_test_binarized, y_pred_prob, average='macro', multi_class='ovr')

    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"Macro AUC:      {auc:.4f}")
    print("Confusion Matrix:\n", cm)

    # ================================
    # 9. Save Metrics & Confusion Matrix
    # ================================
    output_folder = "output/matrix"
    os.makedirs(output_folder, exist_ok=True)

    # Save metrics to a text file
    metrics_file = os.path.join(output_folder, "svm_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Best CV Params: {grid_search.best_params_}\n")
        f.write(f"Best CV Score:  {grid_search.best_score_:.4f}\n")
        f.write("=== Test Set ===\n")
        f.write(f"Accuracy:       {accuracy:.4f}\n")
        f.write(f"Precision:      {precision:.4f}\n")
        f.write(f"Recall:         {recall:.4f}\n")
        f.write(f"F1 Score:       {f:.4f}\n")
        f.write(f"Macro AUC:      {auc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # Save confusion matrix as an image
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.title("SVM Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    cm_image_path = os.path.join(output_folder, "svm_confusion_matrix.png")
    plt.savefig(cm_image_path)
    plt.close()

    # ================================
    # 10. Plot ROC Curves (One-vs-Rest)
    # ================================
    # For multi-class: plot each class's ROC
    # This is optional but shows how to produce the curves.
    fpr = {}
    tpr = {}
    roc_display_labels = ["Negative", "Neutral", "Positive"]

    for i, label in enumerate(classes):
        fpr[label], tpr[label], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    
    plt.figure(figsize=(6, 5))
    for label in classes:
        plt.plot(fpr[label], tpr[label], label=f'ROC of {label}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title("SVM ROC Curves (One-vs-Rest)")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    roc_image_path = os.path.join(output_folder, "svm_roc_curves.png")
    plt.savefig(roc_image_path)
    plt.close()

    print(f"All metrics and plots have been saved to: {output_folder}")

if __name__ == "__main__":
    main()
