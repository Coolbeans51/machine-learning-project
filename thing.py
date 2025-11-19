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

from imblearn.over_sampling import RandomOverSampler

# ========== OPTIONAL MODEL FOR DATASET 4 ==========
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False
    print("[INFO] LightGBM not installed. Dataset 4 will fall back to SVM.")


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
    correct_scores = outputs[torch.arange(len(labels)), labels].unsqueeze(1)
    margins = torch.clamp(outputs - correct_scores + margin, min=0)
    margins[torch.arange(len(labels)), labels] = 0
    return margins.sum() / outputs.size(0)


# --------------------------
# Data loading and preprocessing
# --------------------------
def load_txt_data(data_file, label_file=None):
    data = np.loadtxt(data_file)
    labels = np.loadtxt(label_file, dtype=int) if label_file else None
    return data, labels


def preprocess(train_data, test_data):
    train_data = np.where(train_data > 1e90, np.nan, train_data)
    test_data = np.where(test_data > 1e90, np.nan, test_data)

    col_means = np.nanmean(train_data, axis=0)
    train_data = np.where(np.isnan(train_data), col_means, train_data)
    test_data = np.where(np.isnan(test_data), col_means, test_data)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled


# --------------------------
# Training SVM
# --------------------------
def train_pytorch_svm(train_data, train_labels, num_classes, num_epochs=100, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SVMDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PyTorchSVMClassifier(train_data.shape[1], num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = multiclass_hinge_loss(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/100] Loss: {total_loss/len(loader):.4f}")

    return model


def predict_pytorch(model, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(data).to(device))
        _, preds = torch.max(logits, 1)
    return preds.cpu().numpy()


# --------------------------
# MULTI-DATASET RUNNER
# --------------------------
def run_multi_dataset_classification():
    data_files = sorted(glob.glob("TrainData*.txt"))

    for data_file in data_files:
        idx = int(data_file.split("TrainData")[-1].split(".")[0])
        label_file = f"TrainLabel{idx}.txt"
        test_file = f"TestData{idx}.txt"

        print(f"\n=== Processing dataset #{idx} ===")

        train_data, train_labels = load_txt_data(data_file, label_file)
        test_data, _ = load_txt_data(test_file)
        train_labels = train_labels.astype(int)

        # Debug info
        print(f"[DEBUG] Total samples: {len(train_data)}")
        labels, counts = np.unique(train_labels, return_counts=True)
        print(f"[DEBUG] Label distribution: {dict(zip(labels, counts))}")

        # Convert 1–N labels to 0–(N–1)
        shift = 1 if train_labels.min() == 1 else 0
        if shift:
            train_labels -= 1

        num_classes = train_labels.max() + 1
        print(f"Detected {num_classes} classes")

        # Preprocess
        train_data, test_data = preprocess(train_data, test_data)

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_labels, test_size=0.30, random_state=42, stratify=train_labels
        )

        # Oversample training set ONLY
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        print(f"[INFO] Oversampled training size → {len(X_train)}")

        # ============================================================
        #   SPECIAL CASE: Dataset 4 uses LightGBM instead of SVM
        # ============================================================
        if idx == 4 and LIGHTGBM_AVAILABLE:
            print("[INFO] Using LightGBM for dataset 4")

            lgb_train = lgb.Dataset(X_train, label=y_train)
            params = {
                "objective": "multiclass",
                "num_class": num_classes,
                "learning_rate": 0.05,
                "num_leaves": 64,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
            }

            model = lgb.train(params, lgb_train, num_boost_round=200)

            val_preds = np.argmax(model.predict(X_val), axis=1)

        else:
            # ========================================================
            # DEFAULT: SVM for datasets 1–3
            # ========================================================
            model = train_pytorch_svm(X_train, y_train, num_classes)
            val_preds = predict_pytorch(model, X_val)

        # Metrics
        val_acc = (val_preds == y_val).mean()
        print(f"Validation accuracy: {val_acc:.4f}")

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average="macro", zero_division=0
        )
        print(f"Precision (macro): {prec:.4f}")
        print(f"Recall (macro):    {rec:.4f}")
        print(f"F1-score (macro):  {f1:.4f}")

        # Predict test set
        if idx == 4 and LIGHTGBM_AVAILABLE:
            predictions = np.argmax(model.predict(test_data), axis=1)
        else:
            predictions = predict_pytorch(model, test_data)

        if shift:
            predictions += 1

        np.savetxt(f"Predictions{idx}.txt", predictions, fmt="%d")
        print(f"Saved Predictions{idx}.txt")


# --------------------------
# RUN
# --------------------------
if __name__ == "__main__":
    run_multi_dataset_classification()
