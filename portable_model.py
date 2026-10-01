"""Version-independent copy of the trained model.

A joblib pickle only loads reliably with the scikit-learn version that created it (the notebook used 1.4.2).
The notebook (section 12b) therefore also exports the fitted model as plain numbers (tree arrays / coefficients).
PortableModel re-implements predict_proba from that export using only NumPy. It is used as a fallback when
the pickle cannot be loaded, and is checked against the real model's saved outputs in tests/.
"""
import json
import numpy as np
import pandas as pd


def export_model_to_dict(model, features, threshold, model_name=""):
    """Serialise a fitted LogisticRegression pipeline / RandomForest / GradientBoosting (same format as notebook 12b)."""
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    def tree(t, leaf):
        return {"left": t.children_left.tolist(), "right": t.children_right.tolist(),
                "feature": t.feature.tolist(), "threshold": t.threshold.tolist(), "value": leaf.tolist()}

    out = {"model_name": model_name, "features": list(features), "threshold": float(threshold)}
    if isinstance(model, Pipeline):
        sc, lr = model.named_steps["sc"], model.named_steps["clf"]
        out.update(type="lr", mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
                   coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]))
    elif isinstance(model, RandomForestClassifier):
        out.update(type="rf", trees=[tree(e.tree_, e.tree_.value[:, 0, 1] / e.tree_.value[:, 0, :].sum(axis=1))
                                     for e in model.estimators_])
    elif isinstance(model, GradientBoostingClassifier):
        x0 = np.zeros((1, len(features)))
        contrib = sum(e.tree_.value[e.apply(x0.astype(np.float32))[0], 0, 0] for e in model.estimators_[:, 0])
        init = float(model.decision_function(x0)[0] - model.learning_rate * contrib)
        out.update(type="gb", init=init, learning_rate=float(model.learning_rate),
                   trees=[tree(e.tree_, e.tree_.value[:, 0, 0]) for e in model.estimators_[:, 0]])
    else:
        raise ValueError(f"Unsupported model type for export: {type(model).__name__}")
    return out


def _tree_predict(tr, X32):
    left, right = np.asarray(tr["left"]), np.asarray(tr["right"])
    feat, thr, val = np.asarray(tr["feature"]), np.asarray(tr["threshold"]), np.asarray(tr["value"])
    node = np.zeros(len(X32), dtype=np.int64)
    for _ in range(len(left)):                      # depth can never exceed node count
        is_leaf = left[node] == -1
        if is_leaf.all():
            break
        f = np.where(is_leaf, 0, feat[node])
        go_left = X32[np.arange(len(X32)), f] <= thr[node]
        node = np.where(is_leaf, node, np.where(go_left, left[node], right[node]))
    return val[node]


class PortableModel:
    """NumPy re-implementation of predict_proba for the exported model."""

    def __init__(self, spec):
        self.spec = spec
        self.type = spec["type"]
        self.feature_names_in_ = np.array(spec["features"])
        self.n_features_in_ = len(spec["features"])

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def predict_proba(self, X):
        X = X[list(self.spec["features"])] if isinstance(X, pd.DataFrame) else np.asarray(X)
        A = np.asarray(X, dtype=float)
        if self.type == "lr":
            z = ((A - np.array(self.spec["mean"])) / np.array(self.spec["scale"])) @ np.array(self.spec["coef"]) \
                + self.spec["intercept"]
            p = 1 / (1 + np.exp(-z))
        elif self.type == "rf":
            X32 = A.astype(np.float32)
            p = np.mean([_tree_predict(t, X32) for t in self.spec["trees"]], axis=0)
        elif self.type == "gb":
            X32 = A.astype(np.float32)
            raw = self.spec["init"] + self.spec["learning_rate"] * sum(_tree_predict(t, X32) for t in self.spec["trees"])
            p = 1 / (1 + np.exp(-raw))
        else:
            raise ValueError(f"Unknown model type '{self.type}'.")
        return np.column_stack([1 - p, p])
