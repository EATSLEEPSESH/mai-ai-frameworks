"""Дерево решений CART (своя реализация, векторный поиск разбиений).

Поддерживает две задачи:
  * classification — критерий Джини;
  * regression     — критерий дисперсии (MSE).
"""

import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = value


class DecisionTree:
    def __init__(self, task="classification", max_depth=6, min_samples_leaf=50,
                 max_features=None, random_state=42):
        assert task in ("classification", "regression")
        self.task = task
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

    # ---------- обучение ----------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.n_features_ = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)  # коды классов
        self.root_ = self._build(X, y, np.arange(len(X)), depth=0)
        return self

    def _leaf_value(self, y):
        if self.task == "classification":
            counts = np.bincount(y, minlength=len(self.classes_)).astype(float)
            return counts / counts.sum()
        return float(y.mean())

    def _build(self, X, y, idx, depth):
        node = _Node(self._leaf_value(y[idx]))
        n = len(idx)
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return node
        feat, thr, gain = self._best_split(X, y, idx)
        if feat is None or gain <= 1e-12:
            return node
        mask = X[idx, feat] <= thr
        if mask.all() or not mask.any():
            return node
        node.feature, node.threshold = feat, thr
        node.left = self._build(X, y, idx[mask], depth + 1)
        node.right = self._build(X, y, idx[~mask], depth + 1)
        return node

    def _best_split(self, X, y, idx):
        n = len(idx)
        feats = np.arange(self.n_features_)
        if self.max_features is not None and self.max_features < self.n_features_:
            feats = self.rng.choice(feats, size=self.max_features, replace=False)

        best_gain, best_f, best_thr = 0.0, None, None
        min_leaf = self.min_samples_leaf
        for f in feats:
            x = X[idx, f]
            order = np.argsort(x, kind="mergesort")
            xs = x[order]
            ys = y[idx][order]
            # позиции в отсортированном массиве, где можно резать
            valid = np.nonzero(xs[:-1] < xs[1:])[0] + 1
            valid = valid[(valid >= min_leaf) & (valid <= n - min_leaf)]
            if len(valid) == 0:
                continue
            if self.task == "classification":
                n_classes = len(self.classes_)
                onehot = np.zeros((n, n_classes), dtype=np.float32)
                onehot[np.arange(n), ys] = 1.0
                left_cnt = np.cumsum(onehot, axis=0)[valid - 1]      # кол-во классов слева
                right_cnt = onehot.sum(axis=0)[None, :] - left_cnt
                wl = valid.astype(np.float32)
                wr = (n - valid).astype(np.float32)
                gini_l = 1.0 - ((left_cnt / wl[:, None]) ** 2).sum(axis=1)
                gini_r = 1.0 - ((right_cnt / wr[:, None]) ** 2).sum(axis=1)
                parent = 1.0 - ((onehot.sum(axis=0) / n) ** 2).sum()
                score = parent - (wl * gini_l + wr * gini_r) / n
            else:
                ys = ys.astype(np.float64)
                cs = np.cumsum(ys)
                cs2 = np.cumsum(ys ** 2)
                tot, tot2 = cs[-1], cs2[-1]
                wl = valid.astype(np.float64)
                wr = n - wl
                sum_l, sum_r = cs[valid - 1], tot - cs[valid - 1]
                sq_l, sq_r = cs2[valid - 1], tot2 - cs2[valid - 1]
                var_l = sq_l / wl - (sum_l / wl) ** 2
                var_r = sq_r / wr - (sum_r / wr) ** 2
                parent = tot2 / n - (tot / n) ** 2
                score = parent - (wl * np.clip(var_l, 0, None) +
                                  wr * np.clip(var_r, 0, None)) / n
            j = int(np.argmax(score))
            if score[j] > best_gain:
                best_gain, best_f = float(score[j]), int(f)
                best_thr = float((xs[valid[j] - 1] + xs[valid[j]]) / 2.0)
        return best_f, best_thr, best_gain

    # ---------- предсказание ----------
    def _descend(self, x):
        node = self.root_
        while node.feature is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.task == "classification":
            values = np.array([self._descend(x) for x in X])
            return self.classes_[values.argmax(axis=1)]
        return np.array([self._descend(x) for x in X])

    def predict_proba(self, X):
        """Вероятность класса 1 (для бинарной классификации)."""
        X = np.asarray(X, dtype=np.float32)
        values = np.array([self._descend(x) for x in X])
        return values[:, 1]
