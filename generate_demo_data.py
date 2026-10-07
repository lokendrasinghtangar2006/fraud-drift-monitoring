from pathlib import Path
import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)
    n_samples = 12000
    n_features = 12
    fraud_rate = 0.015
    y = (rng.random(n_samples) < fraud_rate).astype(int)
    X = rng.normal(0, 1, (n_samples, n_features))
    X[:, 0] += y * 2.2
    X[:, 1] -= y * 1.8
    X[:, 2] += y * 1.2
    X[:, 3] += y * rng.normal(0.5, 0.5, n_samples)
    columns = [f"V{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, "Time", np.arange(n_samples) * 30 + rng.integers(0, 10, n_samples))
    df["Amount"] = np.abs(rng.lognormal(mean=3.5, sigma=1.0, size=n_samples))
    df.loc[y == 1, "Amount"] *= rng.uniform(1.2, 3.0, size=np.sum(y == 1))
    df["Class"] = y
    Path("data").mkdir(exist_ok=True)
    output = "data/demo_creditcard.csv"
    df.to_csv(output, index=False)
    print(f"Demo dataset created: {output}")
    print(f"Rows: {len(df)}")
    print(f"Fraud rows: {int(df['Class'].sum())}")
    print(f"Fraud rate: {df['Class'].mean():.4%}")


if __name__ == "__main__":
    main()
