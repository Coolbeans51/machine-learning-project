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


# --------------------------
# Dataset and Model
# --------------------------
class SVMDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]


class PyTorchSVMClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PyTorchSVMClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    def forward(self, x): return self.linear(x)


def multiclass_hinge_loss(outputs, labels, margin=1.0):
    correct_class_scores = outputs[torch.arange(outputs.size(0)), labels].unsqueeze(1)
    margins = torch.clamp(outputs - correct_class_scores + margin, min=0)
    margins[torch.arange(outputs.size(0)), labels] = 0
    return margins.sum() / outputs.size(0)


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
def train_pytorch_svm(train_data, train_labels, num_classes=5, num_epochs=100, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SVMDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = PyTorchSVMClassifier(train_data.shape[1], num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = multiclass_hinge_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(loader):.4f}")

    return model


def predict_pytorch(model, test_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(test_data).to(device))
        _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()


# --------------------------
# Multi-dataset runner with 70/30 validation split
# --------------------------
def run_multi_dataset_classification(use_pytorch=True):
    data_files = sorted(glob.glob("TrainData*.txt"))

    for data_file in data_files:
        idx = data_file.split("TrainData")[-1].split(".")[0]
        label_file = f"TrainLabel{idx}.txt"
        test_file = f"TestData{idx}.txt"

        print(f"\n=== Processing dataset #{idx} ===")

        # Load
        train_data, train_labels = load_txt_data(data_file, label_file)
        test_data, _ = load_txt_data(test_file)

        train_labels = train_labels.astype(int)
        # --------------------  
        # CHECK 1: Sample size  
        # --------------------
        print(f"[DEBUG] Dataset #{idx} sample count: {len(train_data)}")
        if len(train_data) < 200:
            print("[WARNING] Very small dataset. Perfect training accuracy may be normal.")

        # --------------------  
        # CHECK 2: Duplicate rows  
        # --------------------
        unique_rows = np.unique(train_data, axis=0)
        if len(unique_rows) < len(train_data):
            print(f"[WARNING] Dataset #{idx} contains duplicate rows "
                f"({len(train_data) - len(unique_rows)} duplicates). "
                "Perfect training accuracy becomes very likely.")

        # --------------------  
        # CHECK 3: Label distribution  
        # --------------------
        labels, counts = np.unique(train_labels, return_counts=True)
        print(f"[DEBUG] Label distribution: {dict(zip(labels, counts))}")
        if len(labels) == 1:
            print("[ERROR] Dataset has only ONE unique label. Model trivializes and loss becomes zero.")
        elif any(c < 5 for c in counts):
            print("[WARNING] Some classes have extremely low sample counts. "
                "Model may overfit them easily, causing loss=0.0.")


        # Fix label range: convert 1–N → 0–(N–1)
        shift = 0
        if train_labels.min() == 1:
            train_labels -= 1
            shift = 1

        # Dynamic number of classes
        num_classes = int(train_labels.max() + 1)
        print(f"Detected {num_classes} classes (labels 0–{num_classes-1})")

        # Preprocess train+test consistently
        train_data, test_data = preprocess(train_data, test_data)

        # 70/30 split
        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_labels,
            test_size=0.30,
            random_state=42,
            shuffle=True
        )

        print(f"Training size: {len(X_train)} | Validation size: {len(X_val)}")

        # Train + evaluate
        if use_pytorch:
            model = train_pytorch_svm(X_train, y_train, num_classes)

            # Train accuracy
            train_preds = predict_pytorch(model, X_train)
            train_acc = (train_preds == y_train).mean()
            print(f"Training accuracy: {train_acc:.4f}")

            # Validation accuracy
            val_preds = predict_pytorch(model, X_val)
            val_acc = (val_preds == y_val).mean()
            print(f"Validation accuracy: {val_acc:.4f}")

            # Predict test set
            predictions = predict_pytorch(model, test_data)

        else:
            clf = SVC(kernel='rbf')
            clf.fit(X_train, y_train)

            train_acc = clf.score(X_train, y_train)
            val_acc = clf.score(X_val, y_val)
            print(f"Training accuracy: {train_acc:.4f}")
            print(f"Validation accuracy: {val_acc:.4f}")

            predictions = clf.predict(test_data)

        # Shift predictions back if labels originally started at 1
        if shift == 1:
            predictions += 1

        # Save test predictions
        np.savetxt(f"Predictions{idx}.txt", predictions, fmt='%d')
        print(f"Saved Predictions{idx}.txt")


# --------------------------
# Run all datasets
# --------------------------
if __name__ == "__main__":
    run_multi_dataset_classification(use_pytorch=True)
