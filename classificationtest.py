import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

# ====================================================
#  Utilities
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

    vt = VarianceThreshold(threshold=1e-4)
    train_data = vt.fit_transform(train_data)
    test_data = vt.transform(test_data)

    scaler = StandardScaler()
    return scaler.fit_transform(train_data), scaler.transform(test_data)


# ====================================================
#  Main Runner (FIXED)
# ====================================================
def run_multi_dataset_classification():
    accuracies = []

    data_files = sorted(glob.glob("TrainData*.txt"))
    if not data_files:
        print("No training files found.")
        return accuracies

    for data_file in data_files:
        idx = data_file.split("TrainData")[-1].split(".")[0]
        label_file = f"TrainLabel{idx}.txt"
        test_file = f"TestData{idx}.txt"

        print(f"\n{'='*60}")
        print(f"Processing dataset #{idx}")
        print(f"{'='*60}")

        train_data, train_labels = load_txt_data(data_file, label_file)
        test_data, _ = load_txt_data(test_file)

        label_shift = 1 if train_labels.min() == 1 else 0
        if label_shift:
            train_labels -= 1

        train_data, test_data = preprocess(train_data, test_data)

        Cs = [0.1, 1, 10, 100]
        gammas = ['scale', 0.01, 0.001]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        best_acc = 0.0
        best_model = None

        for C in Cs:
            for gamma in gammas:
                fold_accs = []
                for tr, va in skf.split(train_data, train_labels):
                    svm = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
                    svm.fit(train_data[tr], train_labels[tr])
                    preds = svm.predict(train_data[va])
                    fold_accs.append(accuracy_score(train_labels[va], preds))

                mean_acc = np.mean(fold_accs)
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_model = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')

        print(f"[SUMMARY] Dataset {idx} CV Accuracy: {best_acc:.4f}")
        accuracies.append(best_acc)

        best_model.fit(train_data, train_labels)
        test_preds = best_model.predict(test_data)

        if label_shift:
            test_preds += 1

        np.savetxt(f"ThekveliPredictions{idx}.txt", test_preds, fmt='%d')
        print(f"Saved ThekveliPredictions{idx}.txt")

    return accuracies


# ====================================================
#  Entry Point
# ====================================================
if __name__ == '__main__':
    accuracies = run_multi_dataset_classification()

    print("\n" + "="*60)
    print("OVERALL ACCURACY SUMMARY")
    print("="*60)

    if len(accuracies) > 0:
        for i, acc in enumerate(accuracies, start=1):
            print(f"Dataset {i}: CV Accuracy = {acc:.4f}")
        print("-"*60)
        print(f"Average CV Accuracy: {np.mean(accuracies):.4f}")
    else:
        print("No datasets were processed.")