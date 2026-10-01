"""Feature engineering.

compute_features      - features for ONE stock from its holder-count history (inference / dashboard).
                        Row i uses observations <= i only, and is used to predict whether observation i+1 is a decrease.
build_training_table  - reproduces notebook sections 3-4 (the table the model was trained on).
split_chronological   - notebook section 6 (70/15/15 on unique timestamps; no shuffling).
"""
import numpy as np
import pandas as pd
from config import config as C


def compute_features(holders):
    s = pd.Series(np.asarray(holders, dtype=float))
    chg = s.diff(1)
    gr = chg / s.shift(1)
    f = pd.DataFrame({
        "prev_popularity_change": chg,
        "prev_growth_rate": gr,
        "prev_rolling_avg": s.rolling(C.WINDOW).mean(),
        "prev_momentum": s.diff(C.WINDOW),
        "prev_users_holding": s,
        "down_count_7": (chg < 0).astype(float).rolling(C.WINDOW).sum(),
        "growth_vol_7": gr.replace([np.inf, -np.inf], np.nan).rolling(C.WINDOW).std(),
    })
    return f.replace([np.inf, -np.inf], np.nan)


def build_training_table(clean):
    """Exact reproduction of notebook cells 6 and 8. Target `risk` = 1 if popularity_change < 0."""
    g = clean.groupby("name")["users_holding"]
    df = clean.copy()
    df["popularity_change"] = g.diff(1)
    df["growth_rate"] = g.pct_change(1)
    df["rolling_avg"] = g.transform(lambda s: s.rolling(C.WINDOW).mean())
    df["momentum"] = g.diff(C.WINDOW)
    df["risk"] = (df["popularity_change"] < 0).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["popularity_change", "growth_rate", "rolling_avg", "momentum"]).reset_index(drop=True)

    g2 = df.groupby("name")
    df["prev_popularity_change"] = g2["popularity_change"].shift(1)
    df["prev_growth_rate"] = g2["growth_rate"].shift(1)
    df["prev_rolling_avg"] = g2["rolling_avg"].shift(1)
    df["prev_momentum"] = g2["momentum"].shift(1)
    df["prev_users_holding"] = g2["users_holding"].shift(1)
    df["down_count_7"] = g2["risk"].transform(lambda s: s.shift(1).rolling(C.WINDOW).sum())
    df["growth_vol_7"] = g2["growth_rate"].transform(lambda s: s.shift(1).rolling(C.WINDOW).std())
    return df.dropna(subset=C.FEATURES).sort_values("timestamp").reset_index(drop=True)


def split_chronological(data):
    """Train / validation / test by time. Boundaries are the 70 % and 85 % quantiles of UNIQUE timestamps."""
    ts = np.sort(data["timestamp"].unique())
    t1 = ts[int(len(ts) * C.TRAIN_FRAC)]
    t2 = ts[int(len(ts) * (C.TRAIN_FRAC + C.VAL_FRAC))]
    train = data[data.timestamp < t1]
    val = data[(data.timestamp >= t1) & (data.timestamp < t2)]
    test = data[data.timestamp >= t2]
    return train, val, test, pd.Timestamp(t1), pd.Timestamp(t2)
