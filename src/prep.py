"""Подготовка данных: масштабирование, разбиение по времени."""

import numpy as np
import pandas as pd


class StandardScaler:
    """Свой аналог sklearn StandardScaler."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def time_split(df, date_col="date", split_date="2025-01-01"):
    """Хронологическое разбиение: train — до split_date, test — с него."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    mask = df[date_col] < pd.Timestamp(split_date)
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


def to_xy(df, feature_cols, target_col):
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy()
    return X, y
