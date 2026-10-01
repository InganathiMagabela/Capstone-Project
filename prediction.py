"""Model loading (with compatibility checks) and prediction."""
import warnings
import numpy as np
import pandas as pd
from config import config as C
from pipelines.feature_engineering import compute_features
from utils.portable_model import PortableModel
from utils.validation import validate_model_artifact


def load_model(path=None, portable_path=None, allow_fallback=True):
    """Load the saved model artifact. Returns (artifact_dict, status_message).
    Order: joblib pickle -> NumPy fallback built from model_export.json."""
    import joblib
    path = path or C.MODEL_FILE
    portable_path = portable_path or C.PORTABLE_MODEL_FILE
    err = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            art = joblib.load(path)
        rep = validate_model_artifact(art)
        if not rep.ok:
            raise ValueError("; ".join(rep.errors))
        art["source"] = "joblib"
        return art, f"Loaded {type(art['model']).__name__} from {path}."
    except Exception as exc:
        err = exc
    if allow_fallback and PortableModel is not None and str(portable_path) and _exists(portable_path):
        pm = PortableModel.from_json(portable_path)
        art = {"model": pm, "features": list(pm.spec["features"]), "threshold": float(pm.spec["threshold"]),
               "model_name": pm.spec.get("model_name", "unknown"), "source": "portable"}
        rep = validate_model_artifact(art)
        if not rep.ok:
            raise ValueError("Portable model is incompatible: " + "; ".join(rep.errors))
        return art, (f"The pickled model could not be loaded ({type(err).__name__}: {str(err)[:120]}). "
                     "This normally means a different scikit-learn version. Using the equivalent NumPy copy "
                     "(model_export.json) instead.")
    raise ValueError(f"Could not load the model: {err}")


def _exists(p):
    from pathlib import Path
    return Path(p).exists()


def load_model_from_bytes(raw_bytes):
    """Load an UPLOADED joblib file. Security note: joblib files are Python pickles and can run code when
    loaded - only upload models you created."""
    import io, joblib
    art = joblib.load(io.BytesIO(raw_bytes))
    rep = validate_model_artifact(art)
    return art, rep


def predict_proba_checked(art, X):
    """Predict only after confirming the input has exactly the columns the model was trained on."""
    feats = list(art["features"])
    missing = [f for f in feats if f not in X.columns]
    if missing:
        raise ValueError(f"Incompatible input: missing feature(s) {missing}.")
    if X[feats].isna().any(axis=None):
        raise ValueError("Incompatible input: features contain missing values.")
    return art["model"].predict_proba(X[feats])[:, 1]


def label_from_probability(p, threshold):
    return C.HIGH_LABEL if p >= threshold else C.LOW_LABEL


def score_series(df_stock, art):
    """Score every observation of one stock; also records the actual next-observation outcome for a back-test."""
    d = df_stock.sort_values("timestamp").reset_index(drop=True).copy()
    f = compute_features(d["users_holding"].values)
    ok = f[art["features"]].notna().all(axis=1)
    d["probability"] = np.nan
    if ok.any():
        d.loc[ok, "probability"] = predict_proba_checked(art, f.loc[ok])
    d["alert"] = d["probability"] >= art["threshold"]
    nxt = d["users_holding"].shift(-1)
    d["next_decrease"] = np.where(nxt.isna(), np.nan, (nxt < d["users_holding"]).astype(float))
    return d


def predict_from_counts(counts, art):
    """counts: holder counts, oldest first (>= MIN_HISTORY). Returns probability, label and the feature row."""
    counts = [float(c) for c in counts]
    if len(counts) < C.MIN_HISTORY:
        raise ValueError(f"Please provide at least {C.MIN_HISTORY} consecutive observations (you gave {len(counts)}).")
    if any(c < 0 for c in counts):
        raise ValueError("Holder counts cannot be negative.")
    f = compute_features(counts)
    row = f[art["features"]].iloc[[-1]]
    if row.isna().any(axis=None):
        raise ValueError("Not enough valid history: the counts contain zeros or too little variation to compute all features.")
    p = float(predict_proba_checked(art, row)[0])
    return {"probability": p, "label": label_from_probability(p, art["threshold"]),
            "class_probabilities": {C.LOW_LABEL: 1 - p, C.HIGH_LABEL: p},
            "features": row.iloc[0].to_dict()}
