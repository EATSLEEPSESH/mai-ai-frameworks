"""K ближайших соседей: классификация и регрессия (своя реализация на numpy)."""

import numpy as np


class _KNNBase:
    def __init__(self, k=5, weights="uniform", chunk=2048):
        self.k = k
        self.weights = weights
        self.chunk = chunk

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y)
        return self

    def _neighbours(self, Xq):
        """Индексы и расстояния k ближайших соседей (по батчам, через BLAS)."""
        Xq = np.asarray(Xq, dtype=np.float32)
        sq_train = (self.X_ ** 2).sum(axis=1)[None, :]
        for start in range(0, len(Xq), self.chunk):
            q = Xq[start:start + self.chunk]
            # ||x||^2 - 2 x·t + ||t||^2
            d2 = sq_train - 2.0 * (q @ self.X_.T)
            d2 += (q ** 2).sum(axis=1)[:, None]
            np.maximum(d2, 0.0, out=d2)
            idx = np.argpartition(d2, self.k, axis=1)[:, :self.k]
            dist = np.sqrt(np.take_along_axis(d2, idx, axis=1))
            order = np.argsort(dist, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            dist = np.take_along_axis(dist, order, axis=1)
            yield idx, dist

    def _weights(self, dist):
        if self.weights == "uniform":
            return np.ones_like(dist)
        # обратные расстояния; нулевое расстояние -> бесконечный вес
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        zero = dist == 0
        row_zero = zero.any(axis=1)
        if row_zero.any():
            w[row_zero] = zero[row_zero].astype(float)
        return w

    def predict(self, X):
        out = []
        for idx, dist in self._neighbours(X):
            neighbors = self.y_[idx]
            out.append(self._vote(neighbors, self._weights(dist)))
        return np.concatenate(out)

    def _vote(self, neighbors, w):
        raise NotImplementedError


class KNNClassifier(_KNNBase):
    def _vote(self, neighbors, w):
        classes = np.unique(neighbors)
        scores = (neighbors[:, :, None] == classes[None, None, :]) * w[:, :, None]
        return classes[scores.sum(axis=1).argmax(axis=1)]

    def predict_proba(self, X):
        proba = []
        for idx, dist in self._neighbours(X):
            neighbors = self.y_[idx]
            w = self._weights(dist)
            proba.append((w * (neighbors == 1)).sum(axis=1) / w.sum(axis=1))
        return np.concatenate(proba)


class KNNRegressor(_KNNBase):
    def _vote(self, neighbors, w):
        return (neighbors * w).sum(axis=1) / w.sum(axis=1)
