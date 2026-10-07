import json
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-6


def _clean_numeric(series):
    if isinstance(series, np.ndarray):
        values = series.astype(float, copy=False)
        return values[np.isfinite(values)]
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return values.to_numpy(dtype=float)


def psi(reference, current, bins=10):
    reference, current = _clean_numeric(reference), _clean_numeric(current)
    if len(reference) < 2 or len(current) < 2:
        return np.nan
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), EPS, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_level(value):
    if np.isnan(value): return "unknown"
    if value < 0.10: return "low"
    if value < 0.25: return "moderate"
    return "high"


def ks_2sample(x, y):
    """Two-sample KS statistic and asymptotic p-value without scipy/sklearn."""
    x, y = np.sort(_clean_numeric(x)), np.sort(_clean_numeric(y))
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    values = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    d = float(np.max(np.abs(cdf_x - cdf_y)))
    en = np.sqrt(len(x) * len(y) / (len(x) + len(y)))
    lam = (en + 0.12 + 0.11 / en) * d if en > 0 else 0.0
    if lam <= 0:
        p = 1.0
    else:
        terms = [(-1) ** (k - 1) * np.exp(-2 * (k * lam) ** 2) for k in range(1, 101)]
        p = float(np.clip(2 * np.sum(terms), 0, 1))
    return d, p


def feature_drift_report(reference_df, current_df, feature_columns):
    rows = []
    for feature in feature_columns:
        if feature not in reference_df or feature not in current_df:
            continue
        ref, cur = _clean_numeric(reference_df[feature]), _clean_numeric(current_df[feature])
        if len(ref) < 2 or len(cur) < 2:
            continue
        psi_value = psi(ref, cur)
        ks_stat, p_value = ks_2sample(ref, cur)
        rows.append({"feature": feature, "psi": float(psi_value), "psi_level": drift_level(psi_value),
                     "ks_statistic": float(ks_stat), "ks_p_value": float(p_value),
                     "ks_significant": bool(p_value < 0.05)})
    report = pd.DataFrame(rows)
    return report.sort_values("psi", ascending=False).reset_index(drop=True) if not report.empty else report


def confidence_drift_report(reference_scores, current_scores):
    ref, cur = np.asarray(reference_scores, dtype=float), np.asarray(current_scores, dtype=float)
    stat, p = ks_2sample(ref, cur)
    return {"reference_mean": float(np.mean(ref)), "current_mean": float(np.mean(cur)),
            "reference_median": float(np.median(ref)), "current_median": float(np.median(cur)),
            "mean_change": float(np.mean(cur) - np.mean(ref)), "ks_statistic": float(stat),
            "ks_p_value": float(p), "significant": bool(p < 0.05)}


def overall_drift_decision(feature_report, confidence_report, psi_threshold=0.25):
    max_psi = float(feature_report["psi"].max()) if not feature_report.empty else 0.0
    high_count = int((feature_report["psi"] >= psi_threshold).sum()) if not feature_report.empty else 0
    confidence_shift = bool(confidence_report["significant"])
    return {"max_psi": max_psi, "high_drift_feature_count": high_count,
            "confidence_shift_significant": confidence_shift,
            "retraining_required": bool(max_psi >= psi_threshold or high_count >= 2 or confidence_shift)}


def save_report(path, feature_report, confidence_report, decision):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"feature_drift": feature_report.to_dict(orient="records"),
                   "confidence_drift": confidence_report, "decision": decision}, f, indent=2)
