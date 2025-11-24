import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_auc_score
)
import numpy as np

# -----------------------------------------------------------
# 1. MISSING VALUE CLEANING FUNCTION
# -----------------------------------------------------------

def clean_missing(df, text_col=None, label_col=None):
    """Fixes missing text and label values."""
    print("\n--- Missing values BEFORE cleaning ---")
    print(df.isna().sum())

    # Fill missing text with placeholder
    if text_col:
        df[text_col] = df[text_col].fillna("missing_text")

    # Fill missing labels (safe default = 0 / ham)
    if label_col:
        df[label_col] = df[label_col].fillna(0)

    print("\n--- Missing values AFTER cleaning ---")
    print(df.isna().sum())
    return df


# -----------------------------------------------------------
# 2. LOAD AND STANDARDIZE DATA + FIX MISSING VALUES
# -----------------------------------------------------------

# spam_train1.csv → v1=label, v2=text
train1 = pd.read_csv("spam_train1.csv", usecols=["v1", "v2"])
train1.rename(columns={"v1": "label", "v2": "text"}, inplace=True)

# Clean missing values in train1
train1 = clean_missing(train1, text_col="text", label_col="label")

# Convert labels to numeric
train1["numeric_label"] = train1["label"].map({"ham": 0, "spam": 1})
train1["numeric_label"] = train1["numeric_label"].fillna(0).astype(int)

# spam_train2.csv → label, text, label_num
train2 = pd.read_csv("spam_train2.csv", usecols=["label", "text", "label_num"])
train2.rename(columns={"label_num": "numeric_label"}, inplace=True)

# Clean missing values in train2
train2 = clean_missing(train2, text_col="text", label_col="numeric_label")

train2["numeric_label"] = train2["numeric_label"].astype(int)

# spam_test.csv → message
test_data = pd.read_csv("spam_test.csv", usecols=["message"])
test_data.rename(columns={"message": "text"}, inplace=True)

# Clean missing values in test data
test_data = clean_missing(test_data, text_col="text")


# -----------------------------------------------------------
# 3. COMBINE TRAIN DATA
# -----------------------------------------------------------

df_train = pd.concat([
    train1[["text", "numeric_label"]],
    train2[["text", "numeric_label"]]
], ignore_index=True)

X = df_train["text"]
y = df_train["numeric_label"]
X_test = test_data["text"]


# -----------------------------------------------------------
# 4. TRAIN/VALIDATION SPLIT
# -----------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# -----------------------------------------------------------
# 5. PIPELINE: TF-IDF + RANDOM FOREST
# -----------------------------------------------------------

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_df=0.95,
        min_df=3    # You may adjust this later to improve accuracy
    )),
    ("clf", RandomForestClassifier())
])


# -----------------------------------------------------------
# 6. HYPERPARAMETER TUNING FOR RANDOM FOREST
# -----------------------------------------------------------

param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [10, 20, 30],
    "clf__min_samples_split": [5, 10, 15],
    "clf__min_samples_leaf": [2, 4, 6],
    "clf__max_features": ['sqrt','log2'],
    "clf__class_weight": ['balanced']
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)


# -----------------------------------------------------------
# 7. TRAINING ACCURACY
# -----------------------------------------------------------

train_pred = best_model.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print("Training Accuracy:", round(train_acc, 4))


# -----------------------------------------------------------
# 8. VALIDATION METRICS (WITH ROC-AUC)
# -----------------------------------------------------------

y_pred = best_model.predict(X_val)
y_proba = best_model.predict_proba(X_val)[:, 1]

print("\nValidation Accuracy:", round(accuracy_score(y_val, y_pred), 4))
print("Precision:", round(precision_score(y_val, y_pred), 4))
print("Recall:", round(recall_score(y_val, y_pred), 4))
print("F1 Score:", round(f1_score(y_val, y_pred), 4))

roc = roc_auc_score(y_val, y_proba)
print("ROC-AUC:", round(roc, 4))

print("\nClassification Report:\n")
print(classification_report(y_val, y_pred))


# -----------------------------------------------------------
# 9. TRAIN FULL MODEL & PREDICT TEST SET
# -----------------------------------------------------------

best_model.fit(X, y)
test_predictions = best_model.predict(X_test)


# -----------------------------------------------------------
# 10. SAVE OUTPUT FILE
# -----------------------------------------------------------

np.savetxt("YourLastNameSpam.txt", test_predictions, fmt="%d")
print("\nSaved prediction file: YourLastNameSpam.txt")
