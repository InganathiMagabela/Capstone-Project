"""Input validation with plain-language messages.

validate_holdings_data  -> checks the uploaded holder-count CSV (long format: name, timestamp, users_holding)
validate_model_artifact -> checks a loaded model dictionary BEFORE it is allowed to predict
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from config import config as C


@dataclass
class ValidationReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def ok(self):
        return not self.errors


def validate_holdings_data(df):
    """Return a ValidationReport. Errors block the analysis; warnings are cleaned automatically."""
    rep = ValidationReport()
    if df is None or len(df) == 0:
        rep.errors.append("The file is empty or could not be read as a table.")
        return rep

    missing = [c for c in C.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        rep.errors.append("The dataset is missing the required column(s): " + ", ".join(missing) +
                          ". Required columns are: " + ", ".join(C.REQUIRED_COLUMNS) + ".")
        return rep

    rep.info["rows"] = int(len(df))
    # invalid dates
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = int(ts.isna().sum())
    if bad_ts:
        share = bad_ts / len(df)
        (rep.errors if share > 0.5 else rep.warnings).append(
            f"{bad_ts:,} row(s) have a timestamp that cannot be read as a date; these rows will be dropped.")
    # numeric holder counts
    uh = pd.to_numeric(df["users_holding"], errors="coerce")
    bad_num = int(uh.isna().sum())
    if bad_num:
        share = bad_num / len(df)
        (rep.errors if share > 0.5 else rep.warnings).append(
            f"{bad_num:,} row(s) have a missing or non-numeric 'users_holding'; these rows will be dropped.")
    neg = int((uh < 0).sum())
    if neg:
        rep.errors.append(f"{neg:,} row(s) have a negative number of holders, which is not possible.")
    if (uh.dropna() % 1 != 0).any():
        rep.warnings.append("Some holder counts are not whole numbers.")
    # blank names, merge-error rows, duplicates
    blank = int(df["name"].isna().sum())
    if blank:
        rep.warnings.append(f"{blank:,} row(s) have no stock name and will be dropped.")
    merged = int((df["name"] == C.MERGE_ERROR_NAME).sum())
    if merged:
        rep.warnings.append(f"{merged:,} row(s) are labelled '{C.MERGE_ERROR_NAME}' (a merge error in the original "
                            "dataset, notebook section 2); they will be removed.")
    dup = int(df.assign(_t=ts).duplicated(["name", "_t"]).sum())
    if dup:
        rep.warnings.append(f"{dup:,} duplicate (name, timestamp) row(s) will be removed (first kept after sorting).")

    valid = df.assign(_t=ts, _u=uh).dropna(subset=["_t", "_u", "name"])
    valid = valid[valid["name"] != C.MERGE_ERROR_NAME]
    counts = valid.groupby("name").size()
    short = counts[counts < C.MIN_HISTORY]
    if len(short):
        rep.warnings.append(f"{len(short)} stock(s) have fewer than {C.MIN_HISTORY} observations and cannot be scored: "
                            + ", ".join(map(str, short.index[:8])) + ("..." if len(short) > 8 else ""))
    if counts.empty:
        rep.errors.append("No valid rows remain after checking the data.")
    else:
        rep.info.update(stocks=int(counts.size), start=str(valid["_t"].min()), end=str(valid["_t"].max()))
    return rep


def validate_model_artifact(art, features=None):
    """Check a model dictionary {'model','features','threshold',...}. Returns ValidationReport with
    info['compatible'] True/False. Predictions must not be run when errors exist."""
    features = list(features or C.FEATURES)
    rep = ValidationReport()
    if not isinstance(art, dict):
        rep.errors.append("The model file must contain a dictionary with the keys 'model', 'features' and 'threshold'.")
        return rep
    for k in ("model", "features", "threshold"):
        if k not in art:
            rep.errors.append(f"The model file is missing the required entry '{k}'.")
    if rep.errors:
        return rep
    model, art_feats, thr = art["model"], list(art["features"]), art["threshold"]
    if not hasattr(model, "predict_proba"):
        rep.errors.append("The model cannot output probabilities (no predict_proba method).")
    if not isinstance(thr, (int, float, np.floating)) or not (0 < float(thr) < 1):
        rep.errors.append(f"The decision threshold must be a number between 0 and 1 (found {thr!r}).")

    expected_n = getattr(model, "n_features_in_", None)
    if expected_n is None and hasattr(model, "named_steps"):          # Pipeline
        expected_n = getattr(list(model.named_steps.values())[-1], "n_features_in_", None)
    rep.info.update(model_type=type(model).__name__, model_name=art.get("model_name", "unknown"),
                    expected_features=expected_n, provided_features=len(features))
    if expected_n is not None and expected_n != len(features):
        rep.errors.append(f"Model expects {expected_n} features but the application provides {len(features)}.")
    if art_feats != features:
        if sorted(art_feats) == sorted(features):
            rep.errors.append("The model's feature ORDER differs from the application's feature order: "
                              f"model {art_feats} vs application {features}.")
        else:
            rep.errors.append(f"The model's feature names do not match. Model: {art_feats}. Application: {features}.")
    names = getattr(model, "feature_names_in_", None)
    if names is not None and list(names) != features:
        rep.errors.append("The model's stored feature names do not match the application's features.")
    rep.info["compatible"] = not rep.errors
    return rep
