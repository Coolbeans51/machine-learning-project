import os
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
#  Main Runner (Validation-Aware)
# ====================================================

def run_multi_dataset_classification():
    cv_accuracies = []
    train_accuracies = []
    precisions = []
    recalls = []
    f1s = []

    data_files = sorted(glob.glob("TrainData*.txt"))
    if not data_files:
        print("No training files found.")
        return cv_accuracies, train_accuracies, precisions, recalls, f1s

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

        best_val_acc = 0.0
        best_val_metrics = None
        best_model = None

        # Hyperparameter search with validation accuracy tracking
        for C in Cs:
            for gamma in gammas:
                fold_accs = []
                for tr, va in skf.split(train_data, train_labels):
                    svm = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
                    svm.fit(train_data[tr], train_labels[tr])
                    preds = svm.predict(train_data[va])
                    fold_accs.append(accuracy_score(train_labels[va], preds))

                mean_val_acc = np.mean(fold_accs)
                if mean_val_acc > best_val_acc:
                    # Compute precision, recall, f1 on all folds
                    all_preds = []
                    all_true = []
                    for tr, va in skf.split(train_data, train_labels):
                        svm_tmp = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
                        svm_tmp.fit(train_data[tr], train_labels[tr])
                        p = svm_tmp.predict(train_data[va])
                        all_preds.extend(p)
                        all_true.extend(train_labels[va])
                    prec = precision_score(all_true, all_preds, average='macro', zero_division=0)
                    rec = recall_score(all_true, all_preds, average='macro', zero_division=0)
                    f1v = f1_score(all_true, all_preds, average='macro', zero_division=0)

                    best_val_metrics = (prec, rec, f1v)
                    best_val_acc = mean_val_acc
                    best_model = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')

        print(f"[SUMMARY] Dataset {idx} Validation Accuracy (CV): {best_val_acc:.4f}")
        cv_accuracies.append(best_val_acc)
        precisions.append(best_val_metrics[0])
        recalls.append(best_val_metrics[1])
        f1s.append(best_val_metrics[2])

        # Train final model on full training data
        best_model.fit(train_data, train_labels)
        train_preds = best_model.predict(train_data)
        train_acc = accuracy_score(train_labels, train_preds)
        train_accuracies.append(train_acc)
        test_preds = best_model.predict(test_data)

        if label_shift:
            test_preds += 1

        np.savetxt(f"ThekveliPredictions{idx}.txt", test_preds, fmt='%d')
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Precision (macro): {best_val_metrics[0]:.4f}")
        print(f"Recall (macro):    {best_val_metrics[1]:.4f}")
        print(f"F1-score (macro):  {best_val_metrics[2]:.4f}")
        print(f"Saved ThekveliPredictions{idx}.txt")

    return cv_accuracies, train_accuracies, precisions, recalls, f1s


# ====================================================
#  Entry Point
# ====================================================
if __name__ == '__main__':
    cv_accuracies, train_accuracies, precisions, recalls, f1s = run_multi_dataset_classification()

    print("\n" + "="*60)
    print("OVERALL METRIC SUMMARY")
    print("="*60)

    if len(cv_accuracies) > 0:
        for i, acc in enumerate(cv_accuracies, start=1):
            print(f"Dataset {i}: Validation Accuracy = {acc:.4f}")
        print("-"*60)
        print(f"Average Validation Accuracy: {np.mean(cv_accuracies):.4f}")
        print(f"Average Training Accuracy:   {np.mean(train_accuracies):.4f}")
        print(f"Average Precision (macro):  {np.mean(precisions):.4f}")
        print(f"Average Recall (macro):     {np.mean(recalls):.4f}")
        print(f"Average F1-score (macro):   {np.mean(f1s):.4f}")
