"""Случайный лес поверх своего дерева решений (бэггинг + случайные подпространства)."""

import numpy as np

from .tree import DecisionTree


class RandomForest:
    def __init__(self, task="classification", n_trees=40, max_depth=8,
                 min_samples_leaf=30, max_features="sqrt", sample_frac=0.7,
                 random_state=42):
        self.task = task
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.sample_frac = sample_frac
        self.rng = np.random.default_rng(random_state)
        self.trees_ = []

    def _n_features(self, d):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if self.max_features in ("third", "auto"):
            return max(1, d // 3)
        if isinstance(self.max_features, int):
            return self.max_features
        return d

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        n, d = X.shape
        m = self._n_features(d)
        size = max(int(self.sample_frac * n), 2 * self.min_samples_leaf)
        self.trees_ = []
        for _ in range(self.n_trees):
            idx = self.rng.choice(n, size=size, replace=True)
            tree = DecisionTree(task=self.task, max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features=m)
            tree.rng = self.rng  # общий генератор — разные подпространства
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees_], axis=0)

    def predict(self, X):
        if self.task == "classification":
            return (self.predict_proba(X) >= 0.5).astype(int)
        return np.mean([t.predict(X) for t in self.trees_], axis=0)
