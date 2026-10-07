"""Model implementations that avoid compiled scikit-learn estimators.

This project is designed to run on Windows machines where Application Control
may block some scikit-learn native estimator DLLs. The supervised model is a
class-weighted logistic regression trained with NumPy. The unsupervised model
is a lightweight Isolation Forest implementation using NumPy only.
"""

from __future__ import annotations

import math
import joblib
import numpy as np


class StandardScalerLite:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_[~np.isfinite(self.scale_)] = 1.0
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class WeightedLogisticRegression:
    """Binary logistic regression with balanced class weights.

    It is intentionally implemented with NumPy so it does not require a
    compiled scikit-learn estimator. Balanced weighting gives the minority
    fraud class a much larger contribution to the loss than the majority
    legitimate class.
    """

    def __init__(self, learning_rate=0.08, epochs=25, l2=1e-4, random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = 0.0
        self.scaler = StandardScalerLite()

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        Xs = self.scaler.fit_transform(X)

        n, p = Xs.shape
        self.weights_ = np.zeros(p, dtype=float)
        self.bias_ = 0.0

        positives = max(float(np.sum(y == 1)), 1.0)
        negatives = max(float(np.sum(y == 0)), 1.0)
        w_pos = n / (2.0 * positives)
        w_neg = n / (2.0 * negatives)
        sample_weights = np.where(y == 1, w_pos, w_neg)
        sample_weights /= np.mean(sample_weights)

        for epoch in range(self.epochs):
            scores = Xs @ self.weights_ + self.bias_
            probs = self._sigmoid(scores)
            error = (probs - y) * sample_weights

            grad_w = (Xs.T @ error) / n + self.l2 * self.weights_
            grad_b = float(np.mean(error))

            # A small learning-rate decay keeps training stable on the real
            # Credit Card Fraud Detection dataset.
            lr = self.learning_rate / math.sqrt(1.0 + epoch * 0.15)
            self.weights_ -= lr * grad_w
            self.bias_ -= lr * grad_b

        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        p = self._sigmoid(Xs @ self.weights_ + self.bias_)
        return p


class IsolationTree:
    def __init__(self, nodes, max_depth):
        self.max_depth = max_depth
        self.leaf = np.array([n.get("leaf", True) for n in nodes], dtype=bool)
        self.size = np.array([n.get("size", 1) for n in nodes], dtype=np.int32)
        self.feature = np.array([n.get("feature", -1) for n in nodes], dtype=np.int32)
        self.split = np.array([n.get("split", 0.0) for n in nodes], dtype=float)
        self.left = np.array([n.get("left", 0) for n in nodes], dtype=np.int32)
        self.right = np.array([n.get("right", 0) for n in nodes], dtype=np.int32)


class IsolationForestLite:
    """Compact Isolation Forest implementation using NumPy only."""

    def __init__(self, n_estimators=20, max_samples=128, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.trees = []
        self.n_features_ = None
        self.sample_size_ = None
        self.max_depth_ = None

    @staticmethod
    def _c_factor(n):
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        harmonic = np.log(n - 1) + 0.5772156649 + 1.0 / (2 * (n - 1))
        return 2.0 * harmonic - 2.0 * (n - 1) / n

    def _build(self, X, indices, depth, rng):
        size = len(indices)
        leaf = {"leaf": True, "size": int(size)}
        if size <= 1 or depth >= self.max_depth_:
            return [leaf]

        X_node = X[indices]
        mins = np.min(X_node, axis=0)
        maxs = np.max(X_node, axis=0)
        candidates = np.flatnonzero(maxs > mins)
        if len(candidates) == 0:
            return [leaf]

        feature = int(rng.choice(candidates))
        split = float(rng.uniform(float(mins[feature]), float(maxs[feature])))
        left_mask = X_node[:, feature] < split
        right_mask = ~left_mask
        if not left_mask.any() or not right_mask.any():
            return [leaf]

        root = {"leaf": False, "feature": feature, "split": split,
                "left": 1, "right": 0}
        nodes = [root]
        left_start = len(nodes)
        nodes.extend(self._build(X, indices[left_mask], depth + 1, rng))
        right_start = len(nodes)
        nodes.extend(self._build(X, indices[right_mask], depth + 1, rng))
        nodes[0]["left"] = left_start
        nodes[0]["right"] = right_start
        return nodes

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.n_features_ = X.shape[1]
        self.sample_size_ = min(self.max_samples, len(X))
        self.max_depth_ = int(math.ceil(math.log2(max(self.sample_size_, 2))))
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            indices = rng.choice(len(X), size=self.sample_size_, replace=False)
            self.trees.append(IsolationTree(self._build(X, indices, 0, rng), self.max_depth_))
        return self

    def _path_lengths(self, X, tree):
        X = np.asarray(X, dtype=float)
        n = len(X)
        depths = np.zeros(n, dtype=float)
        node_ids = np.zeros(n, dtype=np.int32)
        active = np.ones(n, dtype=bool)

        for _ in range(tree.max_depth + 1):
            if not active.any():
                break
            rows = np.flatnonzero(active)
            ids = node_ids[rows]
            is_leaf = tree.leaf[ids]

            if np.any(is_leaf):
                leaf_rows = rows[is_leaf]
                leaf_ids = ids[is_leaf]
                depths[leaf_rows] += np.array([self._c_factor(int(s)) for s in tree.size[leaf_ids]])
                active[leaf_rows] = False

            branch_rows = rows[~is_leaf]
            if len(branch_rows):
                branch_ids = node_ids[branch_rows]
                features = tree.feature[branch_ids]
                go_left = X[branch_rows, features] < tree.split[branch_ids]
                left_rows = branch_rows[go_left]
                right_rows = branch_rows[~go_left]
                if len(left_rows):
                    node_ids[left_rows] = tree.left[node_ids[left_rows]]
                if len(right_rows):
                    node_ids[right_rows] = tree.right[node_ids[right_rows]]
                depths[branch_rows] += 1.0

        if active.any():
            depths[active] += 1.0
        return depths

    def anomaly_score(self, X):
        if not self.trees:
            raise RuntimeError("IsolationForestLite has not been fitted.")
        X = np.asarray(X, dtype=float)
        mean_path = np.zeros(len(X), dtype=float)
        for tree in self.trees:
            mean_path += self._path_lengths(X, tree)
        mean_path /= len(self.trees)
        c = self._c_factor(self.sample_size_)
        return np.power(2.0, -mean_path / c) if c > 0 else np.zeros(len(X))


class FraudModels:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.supervised = WeightedLogisticRegression(random_state=random_state)
        self.isolation = IsolationForestLite(random_state=random_state)
        self.isolation_scaler = StandardScalerLite()

    def fit_supervised(self, X, y):
        self.supervised.fit(X, y)
        return self

    def supervised_scores(self, X):
        return self.supervised.predict_proba(X)

    def fit_isolation_forest(self, X):
        X_scaled = self.isolation_scaler.fit_transform(X)
        self.isolation.fit(X_scaled)
        return self

    def anomaly_scores(self, X):
        X_scaled = self.isolation_scaler.transform(X)
        return self.isolation.anomaly_score(X_scaled)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
