import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(csv_path="data/creditcard_2023.csv", test_size=0.2, val_size=0.1, seed=42):
    """
    Load and preprocess the Credit Card Fraud Detection Dataset 2023.

    - Drops the `id` column (not a feature).
    - Scales `Amount` with StandardScaler (V1-V28 are already PCA-scaled).
    - Splits into train/val/test with stratification (preserves 50/50 balance).

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test (all np.ndarray, float32)
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["id"])

    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32).reshape(-1, 1)

    amount_idx = df.drop(columns=["Class"]).columns.get_loc("Amount")
    scaler = StandardScaler()

    # First split off test set
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    # Then split remaining into train/val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_ratio, stratify=y_train_full, random_state=seed
    )

    # Fit scaler on train only, apply to all splits (avoid data leakage)
    X_train[:, amount_idx:amount_idx + 1] = scaler.fit_transform(X_train[:, amount_idx:amount_idx + 1])
    X_val[:, amount_idx:amount_idx + 1] = scaler.transform(X_val[:, amount_idx:amount_idx + 1])
    X_test[:, amount_idx:amount_idx + 1] = scaler.transform(X_test[:, amount_idx:amount_idx + 1])

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Train class balance: {np.mean(y_train):.4f}")
    print(f"Val class balance:   {np.mean(y_val):.4f}")
    print(f"Test class balance:  {np.mean(y_test):.4f}")