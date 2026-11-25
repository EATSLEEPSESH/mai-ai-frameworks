"""Построение признаков из сырых выгрузок MOEX.

Акции (TQBR)   -> задача классификации: вырастет ли цена на следующий день.
Облигации (TQCB)-> задача регрессии: доходность к погашению на следующий день.

Все признаки считаются только по прошлым данным (shift/rolling назад),
целевая переменная — будущее значение, поэтому утечки нет.
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")


def _prep(df, price_col="CLOSE"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["TRADEDATE"])
    for c in ("OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "VALUE", "NUMTRADES", "WAPRICE"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df[price_col] > 0]
    df = df.sort_values(["SECID", "date"]).reset_index(drop=True)
    return df


def _rolling(g, col, window):
    return g[col].transform(lambda s: s.rolling(window, min_periods=window).mean())


def build_shares():
    df = _prep(pd.read_csv(os.path.join(RAW, "shares.csv")))
    g = df.groupby("SECID", group_keys=False)

    for lag in (1, 2, 3, 5, 10, 20):
        df[f"ret_{lag}"] = g["CLOSE"].transform(lambda s, l=lag: s.pct_change(l))

    df["vol_5"] = g["ret_1"].transform(lambda s: s.rolling(5, min_periods=5).std())
    df["vol_20"] = g["ret_1"].transform(lambda s: s.rolling(20, min_periods=20).std())

    for w in (5, 20):
        ma = g["CLOSE"].transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
        df[f"ma_ratio_{w}"] = df["CLOSE"] / ma - 1.0

    df["hl_range"] = (df["HIGH"] - df["LOW"]) / df["CLOSE"]
    df["oc_range"] = (df["CLOSE"] - df["OPEN"]) / df["OPEN"].replace(0, np.nan)

    for col, name in (("VOLUME", "vol"), ("VALUE", "val"), ("NUMTRADES", "trd")):
        base = g[col].transform(lambda s: s.rolling(20, min_periods=20).mean())
        df[f"{name}_rel"] = df[col] / base.replace(0, np.nan)

    # RSI(14) по средним приростам/убыткам
    delta = df.groupby("SECID")["CLOSE"].diff()
    up = delta.clip(lower=0).groupby(df["SECID"]).transform(
        lambda s: s.rolling(14, min_periods=14).mean())
    down = (-delta).clip(lower=0).groupby(df["SECID"]).transform(
        lambda s: s.rolling(14, min_periods=14).mean())
    df["rsi_14"] = 100 - 100 / (1 + up / down.replace(0, 1e-9))

    df["weekday"] = df["date"].dt.day_of_week
    df["month"] = df["date"].dt.month

    # рыночный фактор: средняя доходность всех бумаг за день
    market = df.groupby("date")["ret_1"].mean().rename("market_ret_1")
    df = df.merge(market, left_on="date", right_index=True, how="left")

    df["next_close"] = g["CLOSE"].shift(-1)
    df["next_ret"] = df["next_close"] / df["CLOSE"] - 1.0
    df["target_up"] = (df["next_ret"] > 0).astype(int)

    feat = ["ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20", "vol_5", "vol_20",
            "ma_ratio_5", "ma_ratio_20", "hl_range", "oc_range", "vol_rel", "val_rel",
            "trd_rel", "rsi_14", "weekday", "month", "market_ret_1"]
    df = df.dropna(subset=feat + ["next_ret"])
    df[feat].describe().to_csv(os.path.join(OUT, "shares_features_desc.csv"))
    df[["date", "SECID"] + feat + ["next_ret", "target_up"]].to_csv(
        os.path.join(OUT, "shares_features.csv"), index=False)
    print("shares:", df.shape, "признаков:", len(feat))
    return feat


def build_bonds():
    df = _prep(pd.read_csv(os.path.join(RAW, "bonds.csv")))
    df["YIELDCLOSE"] = pd.to_numeric(df["YIELDCLOSE"], errors="coerce")
    df["DURATION"] = pd.to_numeric(df["DURATION"], errors="coerce")
    df["COUPONPERCENT"] = pd.to_numeric(df["COUPONPERCENT"], errors="coerce")
    df["FACEVALUE"] = pd.to_numeric(df["FACEVALUE"], errors="coerce")
    df["MATDATE"] = pd.to_datetime(df["MATDATE"], errors="coerce")
    # доходности вне (0, 60]% — выбросы ISS (мусорные сделки, дефолтные выпуски)
    df = df[df["YIELDCLOSE"].notna() & (df["YIELDCLOSE"] > 0) & (df["YIELDCLOSE"] <= 60)]

    g = df.groupby("SECID", group_keys=False)
    df["yield_lag1"] = g["YIELDCLOSE"].shift(1)
    df["dyield_1"] = df["YIELDCLOSE"] - df["yield_lag1"]
    df["dyield_5"] = df["YIELDCLOSE"] - g["YIELDCLOSE"].shift(5)
    df["dyield_20"] = df["YIELDCLOSE"] - g["YIELDCLOSE"].shift(20)

    df["ret_1"] = g["CLOSE"].transform(lambda s: s.pct_change(1))
    df["ret_5"] = g["CLOSE"].transform(lambda s: s.pct_change(5))
    df["ret_20"] = g["CLOSE"].transform(lambda s: s.pct_change(20))
    df["vol_20"] = g["ret_1"].transform(lambda s: s.rolling(20, min_periods=20).std())

    df["years_to_mat"] = (df["MATDATE"] - df["date"]).dt.days / 365.25
    base_val = g["VALUE"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    base_trd = g["NUMTRADES"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["val_rel"] = df["VALUE"] / base_val.replace(0, np.nan)
    df["trd_rel"] = df["NUMTRADES"] / base_trd.replace(0, np.nan)

    df["price_to_face"] = df["CLOSE"] / df["FACEVALUE"].replace(0, np.nan)
    df["is_ofz"] = df["SECID"].str.startswith("SU").astype(int)

    df["next_yield"] = g["YIELDCLOSE"].shift(-1)
    df["target_next_yield"] = df["next_yield"]

    feat = ["YIELDCLOSE", "dyield_1", "dyield_5", "dyield_20", "ret_1", "ret_5", "ret_20",
            "vol_20", "DURATION", "COUPONPERCENT", "years_to_mat", "val_rel", "trd_rel",
            "price_to_face", "is_ofz"]
    df = df.dropna(subset=feat + ["target_next_yield"])
    df[["date", "SECID"] + feat + ["target_next_yield"]].to_csv(
        os.path.join(OUT, "bonds_features.csv"), index=False)
    print("bonds:", df.shape, "признаков:", len(feat))
    return feat


def main():
    os.makedirs(OUT, exist_ok=True)
    build_shares()
    build_bonds()


if __name__ == "__main__":
    main()
