"""Model evaluation on the chronological TEST period (notebook sections 7, 10, 11)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, classification_report)
from config import config as C
from pipelines.feature_engineering import build_training_table, split_chronological
from pipelines.prediction import predict_proba_checked


def metrics(y, pred, score=None):
    out = {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
           "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0)}
    if score is not None:
        out["ROC_AUC"], out["PR_AUC"] = roc_auc_score(y, score), average_precision_score(y, score)
    return out


def evaluate_model(art, clean):
    """Score the test period with the saved model and the saved threshold. Returns a JSON-friendly dict."""
    data = build_training_table(clean)
    _, _, test, t1, t2 = split_chronological(data)
    y = test["risk"].values
    p = predict_proba_checked(art, test)
    pred = (p >= art["threshold"]).astype(int)
    not_cig = (test["name"] != "CIG").values
    res = {
        "model_name": art.get("model_name", "unknown"), "model_source": art.get("source", "unknown"),
        "threshold": float(art["threshold"]), "test_start": str(t2), "test_rows": int(len(test)),
        "test_positive_rate": float(y.mean()),
        "metrics": metrics(y, pred, p),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),     # [[TN, FP], [FN, TP]]
        "baselines": {
            "always_0": metrics(y, np.zeros(len(y), dtype=int), np.zeros(len(y))),
            "persistence": metrics(y, (test["prev_popularity_change"] < 0).astype(int).values,
                                   (test["prev_popularity_change"] < 0).astype(int).values)},
        "metrics_excluding_CIG": {"test_rows": int(not_cig.sum()), "positive_rate": float(y[not_cig].mean()),
                                  "ROC_AUC": roc_auc_score(y[not_cig], p[not_cig]),
                                  "PR_AUC": average_precision_score(y[not_cig], p[not_cig])},
        "classification_report": classification_report(y, pred, target_names=[C.LOW_LABEL, C.HIGH_LABEL],
                                                        zero_division=0, output_dict=True),
    }
    return res


def permutation_importance_pr_auc(art, X, y, n_repeats=5, seed=C.RANDOM_STATE):
    """Same procedure as notebook section 11: drop in PR-AUC when one feature is shuffled.
    Shows what the model relies on - NOT what causes investors to reduce holdings."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    base = average_precision_score(y, art["model"].predict_proba(X[art["features"]])[:, 1])
    rows = []
    for f in art["features"]:
        drops = []
        for _ in range(n_repeats):
            Xp = X[art["features"]].copy()
            Xp[f] = rng.permutation(Xp[f].values)
            drops.append(base - average_precision_score(y, art["model"].predict_proba(Xp)[:, 1]))
        rows.append((f, float(np.mean(drops)), float(np.std(drops))))
    return pd.DataFrame(rows, columns=["Feature", "Importance", "Std"]).sort_values("Importance", ascending=False)
