import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.dataset import load_data


def find_best_threshold(scores_val, y_val):
    precisions, recalls, thresholds = precision_recall_curve(y_val, scores_val)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def evaluate(y_true, y_pred, scores):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, scores)
    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}


if __name__ == "__main__":
    X_train, X_val, y_val, X_test, y_test = load_data()

    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.00173,
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_train)

    # Use raw anomaly scores (not built-in .predict()), tune threshold like the autoencoder
    scores_val = -iso_forest.score_samples(X_val)
    threshold, best_val_f1 = find_best_threshold(scores_val, y_val)
    print(f"Best threshold (F1-optimized on val): {threshold:.6f} | Val F1: {best_val_f1:.4f}")

    scores_test = -iso_forest.score_samples(X_test)
    y_pred_test = (scores_test > threshold).astype(int)

    test_metrics = evaluate(y_test, y_pred_test, scores_test)
    print(f"Isolation Forest Test metrics (tuned threshold): {test_metrics}")

    # Local Outlier Factor (novelty=True agar bisa predict di data baru)
    lof = LocalOutlierFactor(
        n_neighbors=20,
        novelty=True,
        contamination=0.00173,
        n_jobs=-1,
    )
    lof.fit(X_train)

    scores_val_lof = -lof.score_samples(X_val)
    threshold_lof, best_val_f1_lof = find_best_threshold(scores_val_lof, y_val)
    print(f"\nLOF Best threshold (F1-optimized on val): {threshold_lof:.6f} | Val F1: {best_val_f1_lof:.4f}")

    scores_test_lof = -lof.score_samples(X_test)
    y_pred_test_lof = (scores_test_lof > threshold_lof).astype(int)

    test_metrics_lof = evaluate(y_test, y_pred_test_lof, scores_test_lof)
    print(f"LOF Test metrics (tuned threshold): {test_metrics_lof}")

    # One-Class SVM — subsample training set (RBF kernel is O(n^2)-O(n^3), infeasible on 199k rows)
    rng = np.random.RandomState(42)
    subsample_idx = rng.choice(X_train.shape[0], size=5000, replace=False)
    X_train_sub = X_train[subsample_idx]

    ocsvm = OneClassSVM(kernel="rbf", nu=0.00173, gamma="scale")
    ocsvm.fit(X_train_sub)

    scores_val_svm = -ocsvm.score_samples(X_val)
    threshold_svm, best_val_f1_svm = find_best_threshold(scores_val_svm, y_val)
    print(f"\nOne-Class SVM Best threshold (F1-optimized on val): {threshold_svm:.6f} | Val F1: {best_val_f1_svm:.4f}")

    scores_test_svm = -ocsvm.score_samples(X_test)
    y_pred_test_svm = (scores_test_svm > threshold_svm).astype(int)

    test_metrics_svm = evaluate(y_test, y_pred_test_svm, scores_test_svm)
    print(f"One-Class SVM Test metrics (tuned threshold): {test_metrics_svm}")