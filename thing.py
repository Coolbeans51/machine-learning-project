import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ====================================================
#  Optional library detection
# ====================================================

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False
    print("[WARN] imbalanced-learn not installed.")

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False
    print("[WARN] LightGBM not installed.")


# ====================================================
#  Dataset + PyTorch SVM model
# ====================================================
class SVMDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class PyTorchSVMClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    def forward(self, x):
        return self.linear(x)


def multiclass_hinge_loss(outputs, labels, margin=1.0):
    correct_scores = outputs[torch.arange(outputs.size(0)), labels].unsqueeze(1)
    margins = torch.clamp(outputs - correct_scores + margin, min=0.0)
    margins[torch.arange(outputs.size(0)), labels] = 0.0
    return margins.sum() / outputs.size(0)


def train_pytorch_svm(train_data, train_labels, num_classes, num_epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data.values
    dataset = SVMDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PyTorchSVMClassifier(train_data.shape[1], num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = multiclass_hinge_loss(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0 or epoch in [1, num_epochs]:
            print(f"Epoch [{epoch}/{num_epochs}] Loss: {total_loss/len(loader):.4f}")

    return model


def predict_pytorch(model, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(data, pd.DataFrame):
        data = data.values
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(data).to(device))
        _, preds = torch.max(logits, 1)
    return preds.cpu().numpy()


# ====================================================
#  Loading + Preprocessing
# ====================================================
def load_txt_data(data_file, label_file=None):
    data = np.loadtxt(data_file)
    labels = np.loadtxt(label_file, dtype=int) if label_file else None
    return data, labels


def preprocess(train_data, test_data):
    train_data = np.where(train_data > 1e90, np.nan, train_data)
    test_data = np.where(test_data > 1e90, np.nan, test_data)

    col_means = np.nanmean(train_data, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)

    train_data = np.where(np.isnan(train_data), col_means, train_data)
    test_data = np.where(np.isnan(test_data), col_means, test_data)

    scaler = StandardScaler()
    return scaler.fit_transform(train_data), scaler.transform(test_data)


# ====================================================
#  LightGBM with Cross-Validation (for Dataset 4)
# ====================================================
def train_lgbm_with_cv(X, y, num_classes, feature_names, n_splits=5):
    """
    Train LightGBM with stratified k-fold CV for more reliable estimates.
    Returns the best model and average metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_metrics = []
    best_model = None
    best_val_f1 = -1
    
    # More regularized hyperparameters to reduce overfitting
    lgb_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "learning_rate": 0.03,        # Slower learning
        "num_leaves": 31,             # Reduced complexity (was 64)
        "n_estimators": 300,          # More iterations with slower LR
        "min_child_samples": 15,      # More samples per leaf (was 5)
        "max_depth": 6,               # Limit tree depth
        "reg_alpha": 0.1,             # L1 regularization
        "reg_lambda": 0.1,            # L2 regularization
        "subsample": 0.8,             # Row subsampling
        "colsample_bytree": 0.8,      # Column subsampling
        "verbosity": -1,
        "random_state": 42,
    }
    
    print(f"[INFO] Running {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Apply SMOTE within fold (prevents data leakage)
        if IMBLEARN_AVAILABLE:
            class_counts = np.unique(y_train_fold, return_counts=True)[1]
            if class_counts.min() > 1:
                sampler = SMOTE(k_neighbors=1, random_state=42)
            else:
                sampler = RandomOverSampler(random_state=42)
            X_train_res, y_train_res = sampler.fit_resample(X_train_fold, y_train_fold)
        else:
            X_train_res, y_train_res = X_train_fold, y_train_fold
        
        # Convert to DataFrames
        X_train_df = pd.DataFrame(X_train_res, columns=feature_names)
        X_val_df = pd.DataFrame(X_val_fold, columns=feature_names)
        
        lgb_clf = LGBMClassifier(**lgb_params)
        callbacks = [lgb.early_stopping(stopping_rounds=30)]
        
        lgb_clf.fit(
            X_train_df, y_train_res,
            eval_set=[(X_val_df, y_val_fold)],
            eval_metric="multi_logloss",
            callbacks=callbacks
        )
        
        # Evaluate fold
        val_preds = lgb_clf.predict(X_val_df)
        acc = accuracy_score(y_val_fold, val_preds)
        p, r, f1, _ = precision_recall_fscore_support(
            y_val_fold, val_preds, average="macro", zero_division=0
        )
        
        fold_metrics.append({"acc": acc, "precision": p, "recall": r, "f1": f1})
        print(f"  Fold {fold}: Acc={acc:.4f}, F1={f1:.4f}")
        
        # Track best model by F1
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model = lgb_clf
    
    # Compute average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]
    }
    std_metrics = {
        k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]
    }
    
    print(f"\n[CV Results] Mean ± Std across {n_splits} folds:")
    print(f"  Accuracy:  {avg_metrics['acc']:.4f} ± {std_metrics['acc']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"  Recall:    {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"  F1-score:  {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    
    return best_model, avg_metrics


# ====================================================
#  Main multi-dataset runner
# ====================================================
def run_multi_dataset_classification(use_pytorch=True, use_cv_for_dataset4=True):

    data_files = sorted(glob.glob("TrainData*.txt"))
    if not data_files:
        print("No training files found.")
        return

    for data_file in data_files:

        idx = data_file.split("TrainData")[-1].split(".")[0]
        label_file = f"TrainLabel{idx}.txt"
        test_file = f"TestData{idx}.txt"

        print(f"\n{'='*50}")
        print(f"=== Processing dataset #{idx} ===")
        print(f"{'='*50}")

        train_data, train_labels = load_txt_data(data_file, label_file)
        test_data, _ = load_txt_data(test_file)

        # Convert labels
        train_labels = train_labels.astype(int)
        if train_labels.min() == 1:
            train_labels -= 1
            label_shift = 1
        else:
            label_shift = 0

        num_classes = int(train_labels.max()) + 1
        print(f"Samples: {len(train_labels)}, Classes: {num_classes}")
        
        # Show class distribution
        unique, counts = np.unique(train_labels, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")

        train_data, test_data = preprocess(train_data, test_data)

        # Create consistent feature names
        feature_names = [f"feature_{i}" for i in range(train_data.shape[1])]

        # ======================================================
        #  MODEL TRAINING
        # ======================================================
        trained = None
        uses_lgbm = False

        # ------------------------------------------------------
        #  LightGBM with CV for dataset 4
        # ------------------------------------------------------
        if str(idx) == "4" and LIGHTGBM_AVAILABLE:
            try:
                if use_cv_for_dataset4:
                    trained, avg_metrics = train_lgbm_with_cv(
                        train_data, train_labels, num_classes, feature_names, n_splits=5
                    )
                    uses_lgbm = True
                else:
                    # Fallback to single split (original behavior with better params)
                    X_train, X_val, y_train, y_val = train_test_split(
                        train_data, train_labels,
                        test_size=0.30, stratify=train_labels, random_state=42
                    )
                    
                    if IMBLEARN_AVAILABLE:
                        class_counts = np.unique(y_train, return_counts=True)[1]
                        if class_counts.min() > 1:
                            sampler = SMOTE(k_neighbors=1, random_state=42)
                        else:
                            sampler = RandomOverSampler(random_state=42)
                        X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
                    else:
                        X_train_res, y_train_res = X_train, y_train
                    
                    X_train_df = pd.DataFrame(X_train_res, columns=feature_names)
                    X_val_df = pd.DataFrame(X_val, columns=feature_names)
                    
                    lgb_clf = LGBMClassifier(
                        objective="multiclass",
                        num_class=num_classes,
                        learning_rate=0.03,
                        num_leaves=31,
                        n_estimators=300,
                        min_child_samples=15,
                        max_depth=6,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        verbosity=-1,
                    )
                    
                    callbacks = [lgb.early_stopping(stopping_rounds=30)]
                    lgb_clf.fit(
                        X_train_df, y_train_res,
                        eval_set=[(X_val_df, y_val)],
                        eval_metric="multi_logloss",
                        callbacks=callbacks
                    )
                    
                    trained = lgb_clf
                    uses_lgbm = True
                    
                    # Print metrics
                    train_preds = trained.predict(X_train_df)
                    val_preds = trained.predict(X_val_df)
                    print(f"Training accuracy: {(train_preds == y_train_res).mean():.4f}")
                    print(f"Validation accuracy: {(val_preds == y_val).mean():.4f}")
                    p, r, f, _ = precision_recall_fscore_support(
                        y_val, val_preds, average="macro", zero_division=0
                    )
                    print(f"Precision (macro): {p:.4f}")
                    print(f"Recall (macro):    {r:.4f}")
                    print(f"F1-score (macro):  {f:.4f}")

            except Exception as e:
                print(f"[WARN] LightGBM failed: {e}")
                print("[INFO] Falling back to PyTorch SVM...")

        # ------------------------------------------------------
        #  PyTorch SVM for other datasets or as fallback
        # ------------------------------------------------------
        if trained is None:
            X_train, X_val, y_train, y_val = train_test_split(
                train_data, train_labels,
                test_size=0.30, stratify=train_labels, random_state=42
            )
            
            if IMBLEARN_AVAILABLE:
                sampler = RandomOverSampler(random_state=42)
                X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
                print(f"[INFO] Oversampled: {len(X_train)} → {len(X_train_res)}")
            else:
                X_train_res, y_train_res = X_train, y_train

            model = train_pytorch_svm(X_train_res, y_train_res, num_classes)
            trained = model

            train_preds = predict_pytorch(model, X_train_res)
            val_preds = predict_pytorch(model, X_val)
            
            print(f"Training accuracy: {(train_preds == y_train_res).mean():.4f}")
            print(f"Validation accuracy: {(val_preds == y_val).mean():.4f}")
            p, r, f, _ = precision_recall_fscore_support(
                y_val, val_preds, average="macro", zero_division=0
            )
            print(f"Precision (macro): {p:.4f}")
            print(f"Recall (macro):    {r:.4f}")
            print(f"F1-score (macro):  {f:.4f}")

        # ======================================================
        #  FINAL MODEL: Retrain on ALL data for test predictions
        # ======================================================
        print("\n[INFO] Retraining final model on full dataset...")
        
        if uses_lgbm and LIGHTGBM_AVAILABLE:
            # Apply oversampling to full dataset
            if IMBLEARN_AVAILABLE:
                class_counts = np.unique(train_labels, return_counts=True)[1]
                if class_counts.min() > 1:
                    sampler = SMOTE(k_neighbors=1, random_state=42)
                else:
                    sampler = RandomOverSampler(random_state=42)
                X_full_res, y_full_res = sampler.fit_resample(train_data, train_labels)
            else:
                X_full_res, y_full_res = train_data, train_labels
            
            X_full_df = pd.DataFrame(X_full_res, columns=feature_names)
            
            final_model = LGBMClassifier(
                objective="multiclass",
                num_class=num_classes,
                learning_rate=0.03,
                num_leaves=31,
                n_estimators=150,  # Fixed iterations for final (no early stopping)
                min_child_samples=15,
                max_depth=6,
                reg_alpha=0.1,
                reg_lambda=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
            )
            final_model.fit(X_full_df, y_full_res)
            
            test_df = pd.DataFrame(test_data, columns=feature_names)
            test_preds = final_model.predict(test_df)
        else:
            # PyTorch SVM final model
            if IMBLEARN_AVAILABLE:
                sampler = RandomOverSampler(random_state=42)
                X_full_res, y_full_res = sampler.fit_resample(train_data, train_labels)
            else:
                X_full_res, y_full_res = train_data, train_labels
            
            final_model = train_pytorch_svm(X_full_res, y_full_res, num_classes)
            test_preds = predict_pytorch(final_model, test_data)

        if label_shift == 1:
            test_preds += 1

        np.savetxt(f"Predictions{idx}.txt", test_preds, fmt='%d')
        print(f"Saved Predictions{idx}.txt")


# Run
if __name__ == "__main__":
    run_multi_dataset_classification()