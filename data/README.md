# Dataset folder

Place the Kaggle Credit Card Fraud Detection file here as:

```text
data/creditcard.csv
```

Expected columns:

```text
Time,V1,V2,...,V28,Amount,Class
```

`Class` must be:

- 0 = legitimate
- 1 = fraud

The real dataset is intentionally not included in this repository.

You can test the complete project first with:

```powershell
python generate_demo_data.py
```
