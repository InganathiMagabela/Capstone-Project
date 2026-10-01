"""Controlled re-training that follows notebook sections 6-10 and 12.

 1. chronological split (70/15/15)          2. fit on TRAIN with the tuned hyper-parameters (best_params.json)
 3. choose the decision threshold on VALIDATION (maximum F1)
 4. refit on TRAIN+VALIDATION                5. save model, NumPy copy and metadata

The hyper-parameter SEARCH of notebook section 8 is not repeated here (set tune=True to repeat it).
The final model is the one named in config.DEFAULT_MODEL_NAME - it is NOT re-selected by test-set PR-AUC
(see FINDINGS.md, issue 1).
"""
import json
import numpy as np
import joblib
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import config as C
from pipelines.feature_engineering import build_training_table, split_chronological
from utils.helpers import environment_info, read_json, write_json
from utils.portable_model import export_model_to_dict


def make_model(name, params):
    p = {k: v for k, v in params.items() if k != "cv_pr_auc"}
    if name == "Logistic Regression":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=p["clf__C"],
                                                    random_state=C.RANDOM_STATE))])
    if name == "Random Forest":
        return RandomForestClassifier(class_weight="balanced", random_state=C.RANDOM_STATE, n_jobs=-1, **p)
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=C.RANDOM_STATE, **p)
    raise ValueError(f"Unknown model '{name}'.")


def _fit(model, X, y, name):
    if name == "Gradient Boosting":
        return model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model.fit(X, y)


def best_f1_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    return float(t[np.argmax(f)])


def tune_params(name, X_train, y_train):
    """Notebook section 8 (optional, slow)."""
    Xt, yt = X_train.iloc[::C.TUNE_STEP], y_train.iloc[::C.TUNE_STEP]
    cv = TimeSeriesSplit(n_splits=3)
    grids = {"Logistic Regression": (Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(
                 max_iter=2000, class_weight="balanced", random_state=C.RANDOM_STATE))]),
                 {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}, 6),
             "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=C.RANDOM_STATE, n_jobs=-1),
                 {"n_estimators": [100, 200], "max_depth": [6, 10, 14, None], "min_samples_leaf": [5, 20, 50],
                  "max_features": ["sqrt", 0.5]}, 8),
             "Gradient Boosting": (GradientBoostingClassifier(random_state=C.RANDOM_STATE),
                 {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 4],
                  "subsample": [0.7, 1.0]}, 8)}
    est, grid, n_iter = grids[name]
    s = RandomizedSearchCV(est, grid, n_iter=n_iter, cv=cv, scoring="average_precision",
                           random_state=C.RANDOM_STATE, n_jobs=-1)
    if name == "Gradient Boosting":
        s.fit(Xt, yt, sample_weight=compute_sample_weight("balanced", yt))
    else:
        s.fit(Xt, yt)
    return {**s.best_params_, "cv_pr_auc": float(s.best_score_)}


def train_final_model(clean, model_name=None, out_dir=None, tune=False, params=None):
    """Return the artifact dictionary and write it (plus NumPy copy and metadata) to out_dir."""
    model_name = model_name or C.DEFAULT_MODEL_NAME
    out_dir = out_dir or C.MODELS_DIR
    data = build_training_table(clean)
    train, val, test, t1, t2 = split_chronological(data)
    X_tr, y_tr, X_va, y_va = train[C.FEATURES], train["risk"], val[C.FEATURES], val["risk"]

    if params is None:
        params = tune_params(model_name, X_tr, y_tr) if tune else (read_json(C.BEST_PARAMS_FILE) or {}).get(model_name)
    if params is None:
        raise ValueError(f"No hyper-parameters for '{model_name}'. Run with tune=True or provide best_params.json.")

    # threshold from validation using a model fitted on TRAIN only
    thr_model = _fit(make_model(model_name, params), X_tr, y_tr, model_name)
    threshold = best_f1_threshold(y_va, thr_model.predict_proba(X_va)[:, 1])

    # final model on TRAIN + VALIDATION (test data never touched)
    X_tv, y_tv = np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va])
    import pandas as pd
    X_tv = pd.DataFrame(X_tv, columns=C.FEATURES)
    model = _fit(make_model(model_name, params), X_tv, pd.Series(y_tv), model_name)

    art = {"model": model, "features": list(C.FEATURES), "threshold": float(threshold), "model_name": model_name}
    out_dir = __import__("pathlib").Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, out_dir / "risk_model.joblib")
    write_json(out_dir / "model_export.json", export_model_to_dict(model, C.FEATURES, threshold, model_name))
    write_json(out_dir / "model_metadata.json", {
        "model_name": model_name, "params": params, "features": C.FEATURES, "threshold": threshold,
        "target": "risk = 1 if users_holding decreases at the next observation", "labels": C.LABEL_MAP,
        "split": {"train_end": str(t1), "val_end": str(t2), "train_rows": len(train), "val_rows": len(val),
                  "test_rows": len(test)},
        "random_state": C.RANDOM_STATE, "environment": environment_info()})
    return art
