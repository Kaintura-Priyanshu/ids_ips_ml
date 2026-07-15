"""Load raw flow CSV, clean it, and produce train/test splits + fitted scaler/encoder."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

NON_FEATURE_COLS = {"Label", "Src IP", "Dst IP", "Dst Port"}


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y_raw = df["Label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, df[["Src IP", "Dst IP", "Dst Port"]], test_size=test_size,
        random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_test": y_test,
        "meta_test": meta_test.reset_index(drop=True),
        "scaler": scaler, "label_encoder": le,
        "feature_cols": feature_cols,
    }
