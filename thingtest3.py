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
from sklearn.utils.class_weight import compute_class_weight




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


def multiclass_hinge_loss(outputs, labels, margin=1.0, class_weights=None):
    correct_scores = outputs[torch.arange(outputs.size(0)), labels].unsqueeze(1)
    margins = torch.clamp(outputs - correct_scores + margin, min=0.0)
    margins[torch.arange(outputs.size(0)), labels] = 0.0
    if class_weights is not None:
        sample_weights = class_weights[labels]
        margins = margins * sample_weights.unsqueeze(1)
    return margins.sum() / outputs.size(0)


def train_pytorch_svm(train_data, train_labels, num_classes, num_epochs=100, lr=0.01, use_class_weights=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data.values
    dataset = SVMDataset(train_data, train_labels)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = PyTorchSVMClassifier(train_data.shape[1], num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.3)
    class_weights_tensor = None
    if use_class_weights:
        unique_classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
        class_weights_full = np.ones(num_classes, dtype=np.float32)
        for i, cls in enumerate(unique_classes):
            class_weights_full[cls] = class_weights[i]
            class_weights_tensor = torch.FloatTensor(class_weights_full).to(device)
        print(f"[INFO] Using class weights: {class_weights}")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = multiclass_hinge_loss(outputs, y, class_weights=class_weights_tensor)
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


# Class distribution 

def print_class_distribution(labels, title="Class distribution"):
  unique_classes, counts = np.unique(labels, return_counts=True)
  total_samples = len(labels)
  print(f"\n{title}")
  print(f"{'Class':<10} {'Count':<10} {'Percentage':<15} {'Bar Chart'}")
  max_count = counts.max()
  for cls, count in zip(unique_classes,counts):
    percentage = (count/total_samples) * 100
    bar_length = int((count/max_count) * 30)
    bar = '#' * bar_length
    print(f"{cls:<10} {count:<10} {percentage:<14.2f}% {bar}")
    print(f"Total Samples: {total_samples}")
    print(f"Number of class: {len(unique_classes)}")
    print(f"Min Samples per class: {counts.min()}")
    print(f"Max samples per class: {counts.max()}")
    print(f"Imbalance ratio (max/min): {counts.max() / counts.min():.2f}")
    print()
    return unique_classes,counts
        
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
              
        print_class_distribution (train_labels, "Original Class Distribution")

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

        # ------------------------------------------------------
        #  PyTorch SVM for other datasets or as fallback
        # ------------------------------------------------------
        if trained is None:
            X_train, X_val, y_train, y_val = train_test_split(
                train_data, train_labels,
                test_size=0.30, stratify=train_labels, random_state=42
            )
            
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
        # PyTorch SVM final model
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