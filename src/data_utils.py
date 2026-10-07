from pathlib import Path
import numpy as np
import pandas as pd

TARGET = "Class"
TIME_COL = "Time"


def load_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Put creditcard.csv inside data/ or run generate_demo_data.py."
        )

    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. "
            "Expected a Credit Card Fraud Detection dataset."
        )

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df


def get_feature_columns(df):
    return [c for c in df.columns if c != TARGET]


def split_data(df, test_size=0.25, random_state=42):
    """Stratified train/test split without sklearn."""
    rng = np.random.default_rng(random_state)
    y = df[TARGET].to_numpy(dtype=int)
    train_parts, test_parts = [], []

    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_size)))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X = df[get_feature_columns(df)]
    y_series = df[TARGET]
    return (
        X.iloc[train_idx].copy(), X.iloc[test_idx].copy(),
        y_series.iloc[train_idx].copy(), y_series.iloc[test_idx].copy()
    )


def chronological_windows(df, train_fraction=0.70, current_fraction=0.30):
    if TIME_COL in df.columns:
        ordered = df.sort_values(TIME_COL).reset_index(drop=True)
    else:
        ordered = df.reset_index(drop=True)
    cut = int(len(ordered) * train_fraction)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def basic_summary(df):
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "fraud_count": int(df[TARGET].sum()),
        "legitimate_count": int((df[TARGET] == 0).sum()),
        "fraud_rate": float(df[TARGET].mean()),
        "missing_values": int(df.isna().sum().sum()),
    }
