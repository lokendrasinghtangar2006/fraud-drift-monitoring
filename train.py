import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_utils import load_data, get_feature_columns, split_data, basic_summary
from src.metrics_utils import classification_metrics, find_cost_sensitive_threshold
from src.models import FraudModels


def plot_pr_curve(y_true, scores, path, name):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    positives = max(int(np.sum(y == 1)), 1)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train fraud/anomaly detection models.")
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--fn-cost", type=float, default=10.0)
    parser.add_argument("--fp-cost", type=float, default=1.0)
    args = parser.parse_args()

    df = load_data(args.data)
    features = get_feature_columns(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print("\nDataset summary:")
    print(json.dumps(basic_summary(df), indent=2))

    models = FraudModels()

    print("\nTraining supervised class-weighted Logistic Regression...")
    models.fit_supervised(X_train, y_train)
    train_scores = models.supervised_scores(X_train)
    test_scores = models.supervised_scores(X_test)

    threshold, train_cost = find_cost_sensitive_threshold(
        y_train, train_scores, args.fn_cost, args.fp_cost
    )
    supervised = classification_metrics(y_test, test_scores, threshold)
    supervised.update({"model": "Class-weighted Logistic Regression",
                       "false_negative_cost": args.fn_cost,
                       "false_positive_cost": args.fp_cost,
                       "training_expected_cost": float(train_cost),
                       "test_expected_cost": float(supervised["fn"] * args.fn_cost + supervised["fp"] * args.fp_cost)})

    print("\nSupervised model results:")
    for k, v in supervised.items(): print(f"{k}: {v}")

    print("\nTraining unsupervised Isolation Forest...")
    models.fit_isolation_forest(X_train)
    iso_scores = models.anomaly_scores(X_test)
    iso = classification_metrics(y_test, iso_scores, 0.5)
    iso["model"] = "Isolation Forest"

    print("\nIsolation Forest results:")
    for k, v in iso.items(): print(f"{k}: {v}")

    Path("models").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    models.save("models/fraud_models.joblib")

    with open("models/feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    with open("models/reference_stats.json", "w", encoding="utf-8") as f:
        json.dump({"dataset_path": str(args.data), "rows": len(df),
                   "fraud_rate": float(df["Class"].mean()), "threshold": threshold}, f, indent=2)

    results = {"supervised": supervised, "isolation_forest": iso}
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_pr_curve(y_test, test_scores, "reports/figures/precision_recall_curve.png", "Class-weighted Logistic Regression")
    print("\nSaved model, metrics and PR curve.")


if __name__ == "__main__":
    main()
