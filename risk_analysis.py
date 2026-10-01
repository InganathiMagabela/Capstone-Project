"""Risk summaries for the dashboard: data-quality flags, watchlist, back-test, stock-level descriptors."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from config import config as C
from pipelines.prediction import score_series


def quality_flags(clean):
    """Per-stock data-quality summary (notebook cell 33)."""
    rows = []
    for name, g in clean.groupby("name"):
        s = g.sort_values("timestamp")["users_holding"].astype(float)
        pct = (s.diff() / s.shift(1)).replace([np.inf, -np.inf], np.nan)
        extreme = int((pct.abs() > C.EXTREME_JUMP_PCT).sum())
        zero_share = float((s == 0).mean())
        why = []
        if extreme >= C.EXTREME_JUMP_MIN_COUNT:
            why.append(f"{extreme} jumps of more than 1,000% between consecutive readings")
        if zero_share > C.ZERO_SHARE_MAX:
            why.append(f"{zero_share:.0%} of readings show zero holders")
        rows.append({"name": name, "observations": len(s), "extreme_jumps": extreme,
                     "zero_share": round(zero_share, 4), "data_quality_flag": bool(why), "flag_reason": "; ".join(why)})
    return pd.DataFrame(rows).set_index("name")


def latest_scores(clean, art, tail=40):
    """Latest probability and label for every stock (predicts the NEXT observation)."""
    out = []
    for name, g in clean.groupby("name"):
        sc = score_series(g.sort_values("timestamp").tail(tail), art)
        last = sc.iloc[-1]
        no_p = pd.isna(last["probability"])
        out.append({"name": name, "last_reading": last["timestamp"], "holders": int(last["users_holding"]),
                    "probability_of_decrease": last["probability"],
                    "label": "Insufficient history" if no_p else (C.HIGH_LABEL if last["alert"] else C.LOW_LABEL)})
    return pd.DataFrame(out).set_index("name")


def backtest(scored):
    """Alerts vs the actual next-observation outcome on the scored rows (a descriptive back-test)."""
    x = scored.dropna(subset=["probability", "next_decrease"])
    if x.empty:
        return None
    y, p = x["next_decrease"].astype(int), x["alert"].astype(int)
    return {"observations": int(len(x)), "actual_decreases": int(y.sum()), "alerts": int(p.sum()),
            "alerts_correct": int(((p == 1) & (y == 1)).sum()),
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)), "base_rate": float(y.mean())}


def stock_descriptors(scored, window=500):
    """Simple descriptive (FACTUAL) measures for one stock over the displayed window."""
    v = scored.tail(window)
    s = v["users_holding"].astype(float)
    chg = s.diff()
    return {"latest_holders": int(s.iloc[-1]),
            "share_of_readings_that_fell": float((chg < 0).mean()),
            "net_change_over_window": float(s.iloc[-1] - s.iloc[0]),
            "peak_to_latest_decline": float((s.iloc[-1] / s.max() - 1) if s.max() > 0 else np.nan)}
