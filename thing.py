# thing.py (updated)
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

# imblearn (oversampling)
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
except Exception as e:
    raise ImportError("imblearn is required. Install with `pip install -U imbalanced-learn`.") from e

# optional LightGBM (for dataset 4)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False
    print("[INFO] LightGBM not installed. Dataset 4 will fall back to SVM.")

# --------------------------
# Dataset + Model (PyTorch SVM-like linear classifier)
# --------------------------
class SVMDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

class PyTorchSVMClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
    def forward(self, x): return self.linear(x)

def multiclass_hinge_loss(outputs, labels, margin=1.0):
    # outputs: [B, C], labels: [B]
    correct_scores = outputs[torch.arange(outputs.size(0)), labels].unsqueeze(1)
    margins = torch.clamp(outputs - correct_scores + margin, min=0)
    margins[torch.arange(outputs.size(0)), labels] = 0
    return margins.sum() / outputs.size(0)

# --------------------------
# IO + preprocessing
# --------------------------
def load_txt_data(data_file, label_file=None):
    data = np.loadtxt(data_file)
    labels = np.loadtxt(label_file, dtype=int) if label_file else None
    return data, labels

def preprocess(train_data, test_data):
    # treat huge sentinel as NaN
    train_data = np.where(train_data > 1e90, np.nan, train_data)
    test_data = np.where(test_data > 1e90, np.nan, test_data)

    # column means computed from train (so test uses same imputation)
    col_means = np.nanmean(train_data, axis=0)
    train_data = np.where(np.isnan(train_data), col_means, train_data)
    test_data = np.where(np.isnan(test_data), col_means, test_data)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled

# --------------------------
# Training functions
# --------------------------
def train_pytorch_svm(train_data, train_labels, num_classes, num_epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(loader):.4f}")
    return model

def predict_pytorch(model, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(data).to(device))
        _, preds = torch.max(logits, 1)
    return preds.cpu().numpy()

# --------------------------
# Runner
# --------------------------
def run_multi_dataset_classification():
    data_files = sorted(glob.glob("TrainData*.txt"))
    if not data_files:
        print("No TrainData*.txt files found in current directory.")
        return

    for data_file in data_files:
        # dataset index extractor (keeps int index)
        idx_str = data_file.split("TrainData")[-1].split(".")[0]
        try:
            idx = int(idx_str)
        except:
            idx = idx_str  # fallback if not numeric
        label_file = f"TrainLabel{idx_str}.txt"
        test_file = f"TestData{idx_str}.txt"

        print(f"\n=== Processing dataset #{idx_str} ===")
        train_data, train_labels = load_txt_data(data_file, label_file)
        test_data, _ = load_txt_data(test_file)
        train_labels = train_labels.astype(int)

        # ---------------------------------------------------------------------
        # Basic diagnostics
        # ---------------------------------------------------------------------
        n_samples = len(train_data)
        print(f"[DEBUG] Dataset #{idx_str} sample count: {n_samples}")
        if n_samples < 200:
            print("[WARNING] Very small dataset. Overfitting is likely.")

        unique_rows = np.unique(train_data, axis=0)
        if len(unique_rows) < n_samples:
            print(f"[WARNING] Dataset #{idx_str} contains {n_samples - len(unique_rows)} duplicate row(s).")

        labels_vals, counts = np.unique(train_labels, return_counts=True)
        print(f"[DEBUG] Label distribution (original): {dict(zip(labels_vals, counts))}")
        if any(c < 5 for c in counts):
            print("[WARNING] Some classes have very low sample counts (<5).")

        # shift labels 1..N -> 0..N-1 if required
        shift = 0
        if train_labels.min() == 1:
            train_labels -= 1
            shift = 1

        num_classes = int(train_labels.max() + 1)
        print(f"Detected {num_classes} classes (labels 0–{num_classes-1})")

        # preprocess
        train_data, test_data = preprocess(train_data, test_data)

        # stratified 70/30 split
        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_labels, test_size=0.30, random_state=42, stratify=train_labels
        )
        print(f"Training size: {len(X_train)} | Validation size: {len(X_val)}")

        # Choose oversampler:
        # - Use SMOTE ONLY for dataset 4 (per your request), else RandomOverSampler
        if str(idx_str) == "4":
            try:
                sampler = SMOTE(random_state=42)
                sampler_name = "SMOTE"
            except Exception:
                print("[WARNING] SMOTE not available, falling back to RandomOverSampler")
                sampler = RandomOverSampler(random_state=42)
                sampler_name = "RandomOverSampler"
        else:
            sampler = RandomOverSampler(random_state=42)
            sampler_name = "RandomOverSampler"

        X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
        print(f"[INFO] Applied {sampler_name}: training size {len(X_train)} -> {len(X_train_res)}")
        print(f"[DEBUG] Label distribution (after oversample on train): {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

        # ===========================
        # Train model
        # ===========================
        if str(idx_str) == "4" and LIGHTGBM_AVAILABLE:
            # LightGBM branch for dataset 4
            print("[INFO] Using LightGBM for dataset 4")
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
            model = lgb.train(params, lgb_train, num_boost_round=200)

            # training preds & acc (approx; using model.predict on X_train_res)
            train_preds = np.argmax(model.predict(X_train_res), axis=1)
            train_acc = (train_preds == y_train_res).mean()
            print(f"Training accuracy: {train_acc:.4f}")

            val_preds = np.argmax(model.predict(X_val), axis=1)

        else:
            # SVM (PyTorch linear classifier)
            model = train_pytorch_svm(X_train_res, y_train_res, num_classes)
            train_preds = predict_pytorch(model, X_train_res)
            train_acc = (train_preds == y_train_res).mean()
            print(f"Training accuracy: {train_acc:.4f}")
            val_preds = predict_pytorch(model, X_val)

        # ===========================
        # Metrics (validation)
        # ===========================
        val_acc = (val_preds == y_val).mean()
        print(f"Validation accuracy: {val_acc:.4f}")

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average="macro", zero_division=0
        )
        print(f"Precision (macro): {prec:.4f}")
        print(f"Recall (macro):    {rec:.4f}")
        print(f"F1-score (macro):  {f1:.4f}")

        # predict test set
        if str(idx_str) == "4" and LIGHTGBM_AVAILABLE:
            test_preds = np.argmax(model.predict(test_data), axis=1)
        else:
            test_preds = predict_pytorch(model, test_data)

        # shift predictions back to original label range if needed
        if shift:
            test_preds = test_preds + 1

        # save
        np.savetxt(f"Predictions{idx_str}.txt", test_preds, fmt="%d")
        print(f"Saved Predictions{idx_str}.txt")

if __name__ == "__main__":
    run_multi_dataset_classification()
