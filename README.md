# Fraud/Anomaly Detection with Drift Monitoring

## Project scenario
A fintech company sees transaction patterns change over time because of new fraud tactics, seasonal spending, and customer behaviour. A model trained once and left alone can degrade. This project builds a fraud/anomaly detection system that both flags suspicious transactions and checks whether the model's input/confidence distributions are drifting.

## Requirements covered

- **Extreme class imbalance:** handled with balanced class weighting in the supervised logistic regression loss. This is preferred here to naive random oversampling because it does not duplicate minority observations and is simple to reproduce.
- **Supervised model:** class-weighted Logistic Regression trained with labels.
- **Unsupervised anomaly detector:** a NumPy implementation of **Isolation Forest** that does not use fraud labels while fitting.
- **Comparison:** both models are evaluated against the known fraud labels using Precision, Recall, F1, PR-AUC and ROC-AUC. PR-AUC is emphasized because fraud is highly imbalanced.
- **Drift detection:** PSI and two-sample KS statistics compare a reference/training window with a later window. Model-confidence distribution shift is also checked.
- **Alert threshold:** PSI >= 0.25 or significant confidence shift triggers a retraining recommendation.
- **Cost analysis:** false negatives (missed fraud) and false positives (legitimate transactions blocked) have configurable costs. Default FN cost = 10 and FP cost = 1, so the threshold is chosen to reduce expensive missed fraud while still considering customer friction.
- **Bonus:** `simulate_drift.py` creates a shifted later window so the monitor can be demonstrated locally.

## Important PC compatibility note
This repository is deliberately built for **Windows + VS Code + Python 3.13 + venv** and does **not** require Anaconda, Docker, Oracle/SQL, Node.js, a GPU, or scikit-learn estimators. Some managed Windows machines block native estimator DLLs, so the supervised model, Isolation Forest, metrics, split logic and KS calculation are implemented with NumPy/Pandas/SciPy-compatible code.

## Dataset
Use the Kaggle **Credit Card Fraud Detection** dataset (`creditcard.csv`). Expected columns are:

```text
Time, V1, V2, ..., V28, Amount, Class
```

`Class=0` is legitimate and `Class=1` is fraud.

Put the file at:

```text
data/creditcard.csv
```

Do not commit the CSV to GitHub.

## Setup

Open the VS Code terminal in the project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 1. Test without Kaggle data

Create a small reproducible imbalanced dataset:

```powershell
python generate_demo_data.py
```

Train both models:

```powershell
python train.py --data data/demo_creditcard.csv
```

The default cost assumptions are:

```text
False negative cost = 10
False positive cost = 1
```

Change them if you want:

```powershell
python train.py --data data/demo_creditcard.csv --fn-cost 20 --fp-cost 1
```

## 2. Simulate a later window

```powershell
python simulate_drift.py --input data/demo_creditcard.csv --output data/demo_later_window.csv --strength 0.35
```

Then monitor it:

```powershell
python monitor.py --reference data/demo_creditcard.csv --current data/demo_later_window.csv
```

If PSI crosses the threshold or model confidence shifts significantly, the output says:

```text
DRIFT DETECTED -> RETRAINING RECOMMENDED
```

## 3. Automatic retraining workflow

```powershell
python retrain_if_drift.py --reference data/demo_creditcard.csv --current data/demo_later_window.csv --data data/demo_creditcard.csv
```

## 4. Run on the real Kaggle dataset

Place `creditcard.csv` in `data/`, then:

```powershell
python train.py --data data/creditcard.csv
python simulate_drift.py --input data/creditcard.csv --output data/later_window.csv --strength 0.35
python monitor.py --reference data/creditcard.csv --current data/later_window.csv
```

## Outputs

```text
models/fraud_models.joblib
models/feature_columns.json
models/reference_stats.json
reports/metrics.json
reports/drift_report.json
reports/figures/precision_recall_curve.png
```

## Methodology

### Supervised model
Balanced class-weighted logistic regression uses the known `Class` labels. The class weights are approximately:

```text
w_positive = N / (2 * number_of_fraud_rows)
w_negative = N / (2 * number_of_legitimate_rows)
```

This makes the rare fraud examples matter more during training.

### Isolation Forest
The anomaly detector repeatedly isolates random subsets of transactions using randomly selected features and split points. Transactions requiring shorter average path lengths are more anomalous. Labels are not used to fit this model.

### Precision-Recall
For an imbalanced fraud dataset, accuracy can be misleading because a classifier can appear highly accurate by predicting almost everything as legitimate. Precision, Recall, F1 and especially PR-AUC therefore receive priority.

### Cost-sensitive threshold
A 0.5 probability threshold is not automatically optimal. The project tests thresholds from 0.01 to 0.99 and selects the threshold with the lowest expected cost:

```text
Expected Cost = FN × FN_cost + FP × FP_cost
```

With FN cost 10 and FP cost 1, missing a fraud transaction is treated as ten times more expensive than blocking one legitimate transaction.

### Drift
PSI measures how a feature's distribution has shifted. A common interpretation is:

```text
PSI < 0.10       Low
0.10–0.25        Moderate
>= 0.25          High
```

A two-sample KS statistic/p-value provides a second distribution-shift signal. Model confidence is also compared between windows.


