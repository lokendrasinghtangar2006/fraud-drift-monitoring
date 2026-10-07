import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Check drift and retrain if required."
    )
    parser.add_argument("--reference", default="data/creditcard.csv")
    parser.add_argument("--current", default="data/later_window.csv")
    parser.add_argument("--data", default=None, help="Optional labeled dataset for retraining. If omitted, the reference dataset is used.")
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            "monitor.py",
            "--reference",
            args.reference,
            "--current",
            args.current,
        ],
        check=True,
    )

    report_path = Path("reports/drift_report.json")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    required = report["decision"]["retraining_required"]

    if required:
        print("\nStarting retraining...")
        retrain_data = args.data or args.reference
        print(f"Retraining dataset: {retrain_data}")
        subprocess.run(
            [
                sys.executable,
                "train.py",
                "--data",
                retrain_data,
            ],
            check=True,
        )
        print("\nRetraining completed.")
    else:
        print("\nRetraining not required.")


if __name__ == "__main__":
    main()
