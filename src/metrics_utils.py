import numpy as np


def _binary_counts(y_true, scores, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    return tn, fp, fn, tp


def _precision_recall_f1(y_true, scores, threshold):
    tn, fp, fn, tp = _binary_counts(y_true, scores, threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, tn, fp, fn, tp


def _roc_auc(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0 or neg == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    distinct = np.r_[True, ss[1:] != ss[:-1]]
    tpr = np.r_[0.0, tp[distinct] / pos, 1.0]
    fpr = np.r_[0.0, fp[distinct] / neg, 1.0]
    return float(np.trapezoid(tpr, fpr))


def _average_precision(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = np.sum(y == 1)
    if pos == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    precision = tp / (tp + fp)
    recall = tp / pos
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def classification_metrics(y_true, scores, threshold=0.5):
    precision, recall, f1, tn, fp, fn, tp = _precision_recall_f1(y_true, scores, threshold)
    return {
        "pr_auc": _average_precision(y_true, scores),
        "roc_auc": _roc_auc(y_true, scores),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": float(threshold),
    }


def expected_cost(y_true, scores, threshold, false_negative_cost=10.0, false_positive_cost=1.0):
    _, fp, fn, _ = _binary_counts(y_true, scores, threshold)
    return float(fn * false_negative_cost + fp * false_positive_cost)


def find_cost_sensitive_threshold(y_true, scores, false_negative_cost=10.0, false_positive_cost=1.0, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold, best_cost = 0.5, float("inf")
    for threshold in thresholds:
        cost = expected_cost(y_true, scores, threshold, false_negative_cost, false_positive_cost)
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold, best_cost
