import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# imbalanced-learn imports (SMOTE, RandomOverSampler)
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except Exception as e:
    IMBLEARN_AVAILABLE = False
    SMOTE = None
    RandomOverSampler = None
    print("[ERROR] imbalanced-learn not installed. Install with `pip install -U imbalanced-learn` to use SMOTE/oversampling.")

# Optional LightGBM for dataset 4
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False
    lgb = None
    print("[INFO] LightGBM not installed. Dataset 4 will fall back to SVM if LGBM isn't available.")

# --------------------------
# Dataset + Model
# --------------------------
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
    # outputs: [B, C], labels: [B]
    correct_scores = outputs[torch.arange(outputs.size(0)), labels].unsqueeze(1)
    margins = torch.clamp(outputs - correct_scores + margin, min=0)
    margins[torch.arange(outputs.size(0)), labels] = 0
    return margins.sum() / outputs.size(0)

# --------------------------
# Data loading and preprocessing
# --------------------------
def load_txt_data(data_file, label_file=None):
    data = np.loadtxt(data_file)
    labels = np.loadtxt(label_file, dtype=int) if label_file else None
    return data, labels

def preprocess(train_data, test_data):
    # Convert large sentinel to NaN, fill with column mean, then scale
    train_data = np.where(train_data > 1e90, np.nan, train_data)
    test_data  = np.where(test_data  > 1e90, np.nan, test_data)
    col_means = np.nanmean(train_data, axis=0)
    train_data = np.where(np.isnan(train_data), col_means, train_data)
    test_data  = np.where(np.isnan(test_data), col_means, test_data)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled  = scaler.transform(test_data)
    return train_scaled, test_scaled

# --------------------------
# Training functions
# --------------------------
def train_pytorch_svm(train_data, train_labels, num_classes, num_epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SVMDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = PyTorchSVMClassifier(train_data.shape[1], num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(num_epochs):
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
        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    return model

def predict_pytorch(model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(data).to(device))
        _, preds = torch.max(logits, 1)
    return preds.cpu().numpy()

# --------------------------
# Oversampling helper
# --------------------------
def apply_oversampling(X_train, y_train, dataset_idx, prefer_smote_for_dataset4=True):
    """
    Apply oversampling to X_train/y_train.
    - If dataset_idx == 4 and prefer_smote_for_dataset4 and SMOTE available -> try SMOTE(k_neighbors=1)
    - If SMOTE not possible (e.g. a class has 1 sample) -> fallback to RandomOverSampler
    - For other datasets we use RandomOverSampler (you can change policy here)
    Returns: X_res, y_res, method_name (str)
    """
    if not IMBLEARN_AVAILABLE:
        return X_train, y_train, "none(imblearn_missing)"

    labels, counts = np.unique(y_train, return_counts=True)
    min_count = counts.min()

    # prefer SMOTE for dataset 4 if requested
    if dataset_idx == 4 and prefer_smote_for_dataset4 and SMOTE is not None:
        # SMOTE with k_neighbors=1 requires each class to have at least 2 samples
        if min_count >= 2:
            # enforce k_neighbors=1 per user's request
            sm = SMOTE(random_state=42, k_neighbors=1)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            return X_res, y_res, "SMOTE(k=1)"
        else:
            # can't use SMOTE if any class has only 1 sample in the training split
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X_train, y_train)
            return X_res, y_res, "RandomOverSampler(fallback_due_to_too_small_class)"
    else:
        # default: RandomOverSampler for balance
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train, y_train)
        return X_res, y_res, "RandomOverSampler"

# --------------------------
# Multi-dataset runner
# --------------------------
def run_multi_dataset_classification(use_lightgbm_for_4_if_available=True):
    data_files = sorted(glob.glob("TrainData*.txt"))
    if len(data_files) == 0:
        print("[ERROR] No 'TrainData*.txt' files found in the current directory.")
        return

    for data_file in data_files:
        idx = int(data_file.split("TrainData")[-1].split(".")[0])
        label_file = f"TrainLabel{idx}.txt"
        test_file  = f"TestData{idx}.txt"

        print(f"\n=== Processing dataset #{idx} ===")

        train_data, train_labels = load_txt_data(data_file, label_file)
        test_data, _ = load_txt_data(test_file)
        train_labels = train_labels.astype(int)

        # -----------------------
        # Basic debug checks
        # -----------------------
        n_samples = len(train_data)
        print(f"[DEBUG] Dataset #{idx} sample count: {n_samples}")
        if n_samples < 200:
            print("[WARNING] Very small dataset. Overfitting is likely; treat results cautiously.")

        unique_rows = np.unique(train_data, axis=0)
        if len(unique_rows) < n_samples:
            dup = n_samples - len(unique_rows)
            print(f"[WARNING] Dataset #{idx} contains {dup} duplicate rows. Duplicates can artificially inflate training performance.")

        labels, counts = np.unique(train_labels, return_counts=True)
        print(f"[DEBUG] Label distribution (original): {dict(zip(labels, counts))}")
        if len(labels) == 1:
            print("[ERROR] Only one label present in this dataset; classification is trivial.")
        if any(c < 5 for c in counts):
            print("[WARNING] Some classes have very low sample counts (<5). This increases chance of overfitting or unstable resampling.")

        # shift labels if 1-based
        shift = 0
        if train_labels.min() == 1:
            train_labels -= 1
            shift = 1

        num_classes = int(train_labels.max() + 1)
        print(f"Detected {num_classes} classes (labels 0–{num_classes-1})")

        # Preprocess (fill missing & scale)
        train_data, test_data = preprocess(train_data, test_data)

        # stratified split to keep class proportions in train/val
        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_labels, test_size=0.30, random_state=42, shuffle=True, stratify=train_labels
        )
        print(f"Training size: {len(X_train)} | Validation size: {len(X_val)}")

        # Apply oversampling only to training set:
        X_train_res, y_train_res, method = apply_oversampling(X_train, y_train, idx, prefer_smote_for_dataset4=True)
        print(f"[INFO] Applied oversampling method: {method}. Training size {len(X_train)} -> {len(X_train_res)}")

        # Initialize variables to track which model was used
        model_lgb = None
        model = None
        used_lightgbm = False

        # Train model
        # Use LightGBM for dataset 4 if available & requested
        if idx == 4 and use_lightgbm_for_4_if_available and LIGHTGBM_AVAILABLE and lgb is not None:
            print("[INFO] Training LightGBM on oversampled training set for dataset 4.")
            lgb_train = lgb.Dataset(X_train_res, label=y_train_res)
            params = {
                "objective": "multiclass",
                "num_class": num_classes,
                "learning_rate": 0.05,
                "num_leaves": 64,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "verbosity": -1,
            }
            # try/catch LightGBM training warnings/errors
            try:
                model_lgb = lgb.train(params, lgb_train, num_boost_round=200, verbose_eval=False)
                val_preds = np.argmax(model_lgb.predict(X_val), axis=1)
                used_lightgbm = True
            except Exception as e:
                print("[WARNING] LightGBM training/predict failed; falling back to SVM. Error:", e)
                model = train_pytorch_svm(X_train_res, y_train_res, num_classes)
                val_preds = predict_pytorch(model, X_val)
                used_lightgbm = False
        else:
            # Train the PyTorch SVM
            model = train_pytorch_svm(X_train_res, y_train_res, num_classes)
            train_preds = predict_pytorch(model, X_train_res)
            train_acc = (train_preds == y_train_res).mean()
            print(f"Training accuracy: {train_acc:.4f}")
            val_preds = predict_pytorch(model, X_val)
            used_lightgbm = False

        # Metrics on validation set
        val_acc = (val_preds == y_val).mean()
        print(f"Validation accuracy: {val_acc:.4f}")

        prec, rec, f1, _ = precision_recall_fscore_support(y_val, val_preds, average='macro', zero_division=0)
        print(f"Precision (macro): {prec:.4f}")
        print(f"Recall (macro):    {rec:.4f}")
        print(f"F1-score (macro):  {f1:.4f}")

        # Predict test set using the appropriate model
        if used_lightgbm and model_lgb is not None:
            test_preds = np.argmax(model_lgb.predict(test_data), axis=1)
        else:
            test_preds = predict_pytorch(model, test_data)

        if shift:
            test_preds += 1

        np.savetxt(f"Predictions{idx}.txt", test_preds, fmt='%d')
        print(f"Saved Predictions{idx}.txt")

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    run_multi_dataset_classification()