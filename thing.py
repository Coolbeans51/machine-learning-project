import os
import glob
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# Imbalanced-learn (RandomOverSampler)
try:
    from imblearn.over_sampling import RandomOverSampler
except Exception as e:
    raise ImportError("imblearn is required for RandomOverSampler. Install with `pip install -U imbalanced-learn`.") from e

# Silence the sklearn UndefinedMetricWarning by controlling zero_division in metrics calls;
# still allow other warnings to show
warnings.filterwarnings("default")


# --------------------------
# Dataset and Model
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
        super(PyTorchSVMClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)


def multiclass_hinge_loss(outputs, labels, class_weights=None, margin=1.0):
    """
    Multiclass hinge (Crammer-Singer style) averaged across the batch.
    Optionally applies class_weights (1D tensor of shape [num_classes]) by
    weighting the loss per-sample according to the label's weight.
    """
    # outputs: [batch, C]
    # labels: [batch], long
    batch = outputs.size(0)
    correct_scores = outputs[torch.arange(batch), labels].unsqueeze(1)  # [batch,1]
    margins = torch.clamp(outputs - correct_scores + margin, min=0.0)    # [batch, C]
    margins[torch.arange(batch), labels] = 0.0                         # zero out true class

    per_sample_loss = margins.sum(dim=1)  # [batch]

    if class_weights is not None:
        # class_weights is a 1D tensor shape [C]; weight each sample by class_weights[label]
        sample_weights = class_weights[labels]  # [batch]
        weighted_loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + 1e-12)
        return weighted_loss
    else:
        return per_sample_loss.mean()


# --------------------------
# Data loading and preprocessing
# --------------------------
def load_txt_data(data_file, label_file=None):
    data = np.loadtxt(data_file)
    if label_file:
        labels = np.loadtxt(label_file, dtype=int)
    else:
        labels = None
    return data, labels


def preprocess(train_data, test_data):
    """
    Replace extreme sentinel values (>1e90) with NaN, fill NaN with column means
    computed from train_data, then standard-scale features using StandardScaler.
    """
    train_data = np.where(train_data > 1e90, np.nan, train_data)
    test_data = np.where(test_data > 1e90, np.nan, test_data)

    col_means = np.nanmean(train_data, axis=0)
    train_data = np.where(np.isnan(train_data), col_means, train_data)
    test_data = np.where(np.isnan(test_data), col_means, test_data)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    return train_data, test_data


# --------------------------
# Training functions
# --------------------------
def train_pytorch_svm(train_data, train_labels, num_classes=5, num_epochs=100, lr=0.001, use_class_weights=True):
    """
    Train a simple linear PyTorch classifier with multiclass hinge loss.
    If use_class_weights=True, compute inverse-frequency class weights from train_labels
    (before oversampling) and pass them to the hinge loss so rarer classes get larger weight.
    Note: we assume train_labels are integer 0..C-1
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Optionally compute class weights from the *original* (pre-oversample) training labels
    class_weights_tensor = None
    if use_class_weights:
        # Build weights: inverse frequency (smaller freq -> larger weight)
        labels_unique, counts = np.unique(train_labels, return_counts=True)
        inv_freq = np.zeros(int(num_classes), dtype=float)
        # Avoid division by zero
        for lab, cnt in zip(labels_unique, counts):
            inv_freq[int(lab)] = 1.0 / (cnt + 1e-12)
        # Normalize so average weight = 1.0 (optional but keeps magnitudes stable)
        inv_freq = inv_freq / np.mean(inv_freq[inv_freq > 0])
        class_weights_tensor = torch.FloatTensor(inv_freq).to(device)

    dataset = SVMDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = PyTorchSVMClassifier(train_data.shape[1], num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)  # [batch, C]
            loss = multiclass_hinge_loss(outputs, y_batch, class_weights=class_weights_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg_loss = total_loss / (n_batches + 1e-12)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

    return model


def predict_pytorch(model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(data).to(device))
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()


# --------------------------
# Multi-dataset runner with stratified 70/30 and RandomOverSampler on training set
# --------------------------
def run_multi_dataset_classification(use_pytorch=True, apply_oversample=True):
    data_files = sorted(glob.glob("TrainData*.txt"))

    for data_file in data_files:
        idx = data_file.split("TrainData")[-1].split(".")[0]
        label_file = f"TrainLabel{idx}.txt"
        test_file = f"TestData{idx}.txt"

        print(f"\n=== Processing dataset #{idx} ===")

        # Load files
        train_data, train_labels = load_txt_data(data_file, label_file)
        test_data, _ = load_txt_data(test_file)

        # Basic validation
        if train_labels is None:
            print(f"[ERROR] Missing labels for {data_file} - skipping")
            continue

        train_labels = train_labels.astype(int)

        # --------------------
        # DEBUG CHECKS
        # --------------------
        print(f"[DEBUG] Dataset #{idx} sample count: {len(train_data)}")
        if len(train_data) < 200:
            print("[WARNING] Very small dataset. Overfitting is very likely with small sample counts.")

        unique_rows = np.unique(train_data, axis=0)
        if len(unique_rows) < len(train_data):
            dup = len(train_data) - len(unique_rows)
            print(f"[WARNING] Dataset #{idx} contains {dup} duplicate rows. Duplicates can inflate train performance.")

        labels, counts = np.unique(train_labels, return_counts=True)
        print(f"[DEBUG] Label distribution (original): {dict(zip(labels, counts))}")
        if len(labels) == 1:
            print("[ERROR] Dataset has only ONE unique label. Cannot train a multi-class classifier.")
            continue
        if any(c < 5 for c in counts):
            print("[WARNING] Some classes have very low sample counts (<5). This increases overfitting risk.")

        # Normalize label range (convert 1–N → 0–(N–1) if needed)
        shift = 0
        if train_labels.min() == 1:
            train_labels -= 1
            shift = 1

        num_classes = int(train_labels.max() + 1)
        print(f"Detected {num_classes} classes (labels 0–{num_classes-1})")

        # Preprocess (fill missing, scale)
        train_data, test_data = preprocess(train_data, test_data)

        # Stratified 70/30 split to preserve label distribution in validation
        X_train, X_val, y_train, y_val = train_test_split(
            train_data,
            train_labels,
            test_size=0.30,
            random_state=42,
            shuffle=True,
            stratify=train_labels  # stratified split
        )

        print(f"Training size: {len(X_train)} | Validation size: {len(X_val)}")

        # Apply RandomOverSampler on training data ONLY to fix imbalance
        if apply_oversample:
            ros = RandomOverSampler(random_state=42)
            X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
            print(f"[INFO] Applied RandomOverSampler: training size {len(X_train)} -> {len(X_train_res)}")
        else:
            X_train_res, y_train_res = X_train, y_train

        # Show new class distribution after oversample
        lab_r, cnt_r = np.unique(y_train_res, return_counts=True)
        print(f"[DEBUG] Label distribution (after oversample on train): {dict(zip(lab_r, cnt_r))}")

        # -------------------------
        # Train + evaluate
        # -------------------------
        if use_pytorch:
            # Use class-weighting in loss computed from original (pre-oversample) class frequencies,
            # but since we oversampled, class weights will be closer to 1; still useful as an option.
            model = train_pytorch_svm(X_train_res, y_train_res, num_classes=num_classes,
                                      num_epochs=100, lr=0.001, use_class_weights=True)

            # Training accuracy (on oversampled train set)
            train_preds = predict_pytorch(model, X_train_res)
            train_acc = (train_preds == y_train_res).mean()
            print(f"Training accuracy: {train_acc:.4f}")

            # Validation accuracy (on held-out stratified fold)
            val_preds = predict_pytorch(model, X_val)
            val_acc = (val_preds == y_val).mean()
            print(f"Validation accuracy: {val_acc:.4f}")

            # Precision / Recall / F1 (macro). zero_division=0 prevents warnings from classes with no predicted samples
            prec, rec, f1, _ = precision_recall_fscore_support(y_val, val_preds, average='macro', zero_division=0)
            print(f"Precision (macro): {prec:.4f}")
            print(f"Recall (macro):    {rec:.4f}")
            print(f"F1-score (macro):  {f1:.4f}")

            # Final predictions on test set
            predictions = predict_pytorch(model, test_data)

        else:
            # sklearn SVC with class_weight to help with imbalance (in addition to oversampling)
            clf = SVC(kernel="rbf", class_weight='balanced', probability=False)
            clf.fit(X_train_res, y_train_res)

            train_acc = clf.score(X_train_res, y_train_res)
            val_acc = clf.score(X_val, y_val)
            print(f"Training accuracy: {train_acc:.4f}")
            print(f"Validation accuracy: {val_acc:.4f}")

            val_preds = clf.predict(X_val)
            prec, rec, f1, _ = precision_recall_fscore_support(y_val, val_preds, average='macro', zero_division=0)
            print(f"Precision (macro): {prec:.4f}")
            print(f"Recall (macro):    {rec:.4f}")
            print(f"F1-score (macro):  {f1:.4f}")

            predictions = clf.predict(test_data)

        # Shift predictions back to original label range if needed (1..N)
        if shift == 1:
            predictions = predictions + 1

        # Save predictions
        out_file = f"Predictions{idx}.txt"
        np.savetxt(out_file, predictions, fmt='%d')
        print(f"Saved {out_file}\n")


# --------------------------
# Run all datasets
# --------------------------
if __name__ == "__main__":
    run_multi_dataset_classification(use_pytorch=True, apply_oversample=True)
