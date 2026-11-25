"""Градиентный бустинг на своих деревьях регрессии."""

import numpy as np

from .tree import DecisionTree


class GradientBoosting:
    """Бустинг:
      * task='regression'      — квадратичная функция потерь, y_hat = base + Σ lr·f_m;
      * task='classification'  — логистическая потеря, работаем в логитах.
    """

    def __init__(self, task="regression", n_estimators=200, learning_rate=0.1,
                 max_depth=3, min_samples_leaf=30, subsample=1.0, random_state=42):
        self.task = task
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float)
        n = len(X)
        if self.task == "classification":
            p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            self.base_ = float(np.log(p0 / (1 - p0)))
            raw = np.full(n, self.base_)
        else:
            self.base_ = float(y.mean())
            raw = np.full(n, self.base_)

        self.trees_ = []
        for m in range(self.n_estimators):
            if self.task == "classification":
                p = 1.0 / (1.0 + np.exp(-raw))
                resid = y - p
            else:
                resid = y - raw
            if self.subsample < 1.0:
                idx = self.rng.choice(n, size=int(self.subsample * n), replace=False)
            else:
                idx = np.arange(n)
            tree = DecisionTree(task="regression", max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf)
            tree.rng = self.rng
            tree.fit(X[idx], resid[idx])
            upd = tree.predict(X)
            if self.task == "classification":
                # шаг Ньютона с постоянной кривизной (максимум p(1-p) = 0.25)
                raw += self.lr * upd / 0.25
            else:
                raw += self.lr * upd
            self.trees_.append(tree)
        return self

    def _raw(self, X):
        X = np.asarray(X, dtype=np.float32)
        raw = np.full(len(X), self.base_)
        for tree in self.trees_:
            upd = tree.predict(X)
            if self.task == "classification":
                raw += self.lr * upd / 0.25  # усреднённая кривизна в корне
            else:
                raw += self.lr * upd
        return raw

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-self._raw(X)))

    def predict(self, X):
        if self.task == "classification":
            return (self.predict_proba(X) >= 0.5).astype(int)
        return self._raw(X)
