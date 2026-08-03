import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(csv_path="data/creditcard.csv", val_size=0.15, test_size=0.15, seed=42):
    df = pd.read_csv(csv_path)

    # Log-transform Amount (heavily right-skewed) before scaling
    df["Amount"] = np.log1p(df["Amount"])

    # Scale Time and Amount (V1-V28 already PCA-transformed/scaled)
    scaler = StandardScaler()
    df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    val_ratio = val_size / (1 - test_size)
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed
    )

    X_train = X_train_full[y_train_full == 0]

    return X_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    X_train, X_val, y_val, X_test, y_test = load_data()
    print(f"Train (normal only): {X_train.shape}")
    print(f"Val: {X_val.shape}, fraud ratio: {y_val.mean():.5f}")
    print(f"Test: {X_test.shape}, fraud ratio: {y_test.mean():.5f}")