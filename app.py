"""Optional lightweight report viewer.
Run: python app.py
"""
import json
from pathlib import Path


if __name__ == "__main__":
    metrics = Path("reports/metrics.json")
    drift = Path("reports/drift_report.json")
    print("Fraud/Anomaly Detection with Drift Monitoring")
    print("=" * 50)
    if metrics.exists():
        print("\nModel results:")
        print(json.dumps(json.loads(metrics.read_text(encoding="utf-8")), indent=2))
    else:
        print("Run train.py first.")
    if drift.exists():
        print("\nDrift report:")
        print(json.dumps(json.loads(drift.read_text(encoding="utf-8")), indent=2))
    else:
        print("Run monitor.py to create a drift report.")
