import argparse
from pathlib import Path
import pandas as pd

from src.data_utils import load_data, get_feature_columns
from src.models import FraudModels
from src.drift import feature_drift_report, confidence_drift_report, overall_drift_decision, save_report


def main():
    parser = argparse.ArgumentParser(description="Monitor fraud model drift.")
    parser.add_argument("--reference", default="data/creditcard.csv")
    parser.add_argument("--current", default="data/later_window.csv")
    parser.add_argument("--model", default="models/fraud_models.joblib")
    parser.add_argument("--psi-threshold", type=float, default=0.25)
    args = parser.parse_args()

    reference = load_data(args.reference)
    current = load_data(args.current)
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run train.py first.")

    models = FraudModels.load(model_path)
    features = [c for c in get_feature_columns(reference) if c in current.columns]

    # Confidence monitoring is computed on a capped sample so the monitor
    # remains responsive on the large Kaggle dataset.
    ref_sample = reference[features].sample(min(len(reference), 20000), random_state=42)
    cur_sample = current[features].sample(min(len(current), 20000), random_state=42)
    reference_scores = models.supervised_scores(ref_sample)
    current_scores = models.supervised_scores(cur_sample)

    feature_report = feature_drift_report(reference, current, features)
    confidence_report = confidence_drift_report(reference_scores, current_scores)
    decision = overall_drift_decision(feature_report, confidence_report, args.psi_threshold)

    save_report("reports/drift_report.json", feature_report, confidence_report, decision)

    print("\n=== DRIFT MONITORING REPORT ===")
    print(f"Maximum PSI: {decision['max_psi']:.4f}")
    print(f"High-drift features: {decision['high_drift_feature_count']}")
    print(f"Model confidence shift significant: {decision['confidence_shift_significant']}")
    print("\nTop drifted features:")
    if feature_report.empty:
        print("No numeric features available.")
    else:
        print(feature_report[["feature", "psi", "psi_level", "ks_statistic", "ks_p_value"]].head(10).to_string(index=False))

    if decision["retraining_required"]:
        print("\nDRIFT DETECTED -> RETRAINING RECOMMENDED")
    else:
        print("\nNO SIGNIFICANT DRIFT -> KEEP CURRENT MODEL")
    print("\nSaved: reports/drift_report.json")


if __name__ == "__main__":
    main()
