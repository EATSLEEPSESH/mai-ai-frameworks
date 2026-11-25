"""Линейная и логистическая регрессия (своя реализация, градиентный спуск)."""

import numpy as np


class LinearRegressionGD:
    """Линейная регрессия с L2-регуляризацией, полный градиентный спуск.

    Признаки должны быть стандартизованы — тогда один lr годится для всех.
    """

    def __init__(self, lr=0.1, epochs=300, l2=0.0, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.verbose = verbose

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = float(y.mean())
        for epoch in range(self.epochs):
            pred = X @ self.w_ + self.b_
            err = pred - y
            grad_w = X.T @ err / n + self.l2 * self.w_
            grad_b = err.mean()
            self.w_ -= self.lr * grad_w
            self.b_ -= self.lr * grad_b
            if self.verbose and (epoch + 1) % 50 == 0:
                mse = float(np.mean(err ** 2))
                print(f"  epoch {epoch + 1}: MSE={mse:.4f}")
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w_ + self.b_


class RidgeRegressionClosedForm:
    """Замкнутое решение ridge-регрессии: w = (X^T X + λI)^-1 X^T y."""

    def __init__(self, l2=1.0):
        self.l2 = l2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        Xb = np.hstack([np.ones((n, 1)), X])
        reg = self.l2 * np.eye(d + 1)
        reg[0, 0] = 0.0  # свободный член не регуляризуем
        A = Xb.T @ Xb + reg
        try:
            self.coef_ = np.linalg.solve(A, Xb.T @ y)
        except np.linalg.LinAlgError:
            # λ = 0 и мультиколлинеарность признаков: переходим к псевдорешению
            self.coef_ = np.linalg.lstsq(A, Xb.T @ y, rcond=None)[0]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.hstack([np.ones((len(X), 1)), X]) @ self.coef_


class LogisticRegressionGD:
    """Логистическая регрессия с L2, полный градиентный спуск."""

    def __init__(self, lr=0.5, epochs=400, l2=1e-3, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.verbose = verbose

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        for epoch in range(self.epochs):
            p = self._sigmoid(X @ self.w_ + self.b_)
            err = p - y
            self.w_ -= self.lr * (X.T @ err / n + self.l2 * self.w_)
            self.b_ -= self.lr * err.mean()
            # защита от разгона весов при большом шаге: без неё логит уходит
            # в бесконечность и в градиенте появляется NaN
            np.clip(self.w_, -100.0, 100.0, out=self.w_)
            self.b_ = float(np.clip(self.b_, -100.0, 100.0))
            if self.verbose and (epoch + 1) % 100 == 0:
                loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
                print(f"  epoch {epoch + 1}: logloss={loss:.4f}")
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.asarray(X, dtype=float) @ self.w_ + self.b_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
