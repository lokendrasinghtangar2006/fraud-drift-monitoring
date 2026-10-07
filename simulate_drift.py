import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Simulate a shifted later data window.")
    parser.add_argument("--input", default="data/creditcard.csv")
    parser.add_argument("--output", default="data/later_window.csv")
    parser.add_argument("--strength", type=float, default=0.35)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    rng = np.random.default_rng(42)

    if "Amount" in df.columns:
        amount = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
        scale = amount.std() if amount.std() > 0 else 1.0
        df["Amount"] = np.maximum(0, amount * (1 + args.strength) + rng.normal(0, 0.10 * scale, len(df)))

    candidate_features = [c for c in df.columns if c.startswith("V")]
    for feature in candidate_features[:3]:
        values = pd.to_numeric(df[feature], errors="coerce")
        scale = values.std() if values.std() > 0 else 1.0
        df[feature] = values + rng.normal(0, args.strength * 0.20 * scale, len(df))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Simulated later window saved to: {args.output}")
    print(f"Drift strength: {args.strength}")


if __name__ == "__main__":
    main()
