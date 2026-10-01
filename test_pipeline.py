"""Run:  python -m pytest tests -q     (or: python tests/test_pipeline.py)
Tests use the bundled Capstone data/model. The retraining test is slow (~30 s) and skipped unless RUN_SLOW=1."""
import io, os, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import config as C
from pipelines.preprocessing import clean_raw, load_and_validate
from pipelines.feature_engineering import build_training_table, compute_features, split_chronological
from pipelines.prediction import load_model, predict_from_counts, score_series, predict_proba_checked
from pipelines.evaluation import evaluate_model
from utils.validation import validate_holdings_data, validate_model_artifact
from utils.portable_model import PortableModel, export_model_to_dict

_cache = {}


def clean():
    if "c" not in _cache:
        _cache["c"], _ = load_and_validate(C.CLEAN_DATA_FILE)
    return _cache["c"]


def art():
    if "a" not in _cache:
        _cache["a"], _ = load_model()
    return _cache["a"]


# ---------------- data validation
def test_missing_column_message():
    rep = validate_holdings_data(pd.DataFrame({"name": ["A"], "timestamp": ["2020-01-01"]}))
    assert not rep.ok and "users_holding" in rep.errors[0]


def test_bad_values_flagged():
    df = pd.DataFrame({"name": ["A", "A", "A", "robin_dataset"],
                       "timestamp": ["2020-01-01", "2020-01-01", "not a date", "2020-01-02"],
                       "users_holding": [5, 5, 7, -3]})
    rep = validate_holdings_data(df)
    assert not rep.ok and any("negative" in e for e in rep.errors)
    text = " ".join(rep.warnings)
    assert "duplicate" in text and "cannot be read as a date" in text and "robin_dataset" in text


def test_empty_and_unreadable():
    assert not validate_holdings_data(pd.DataFrame()).ok
    clean_df, rep = load_and_validate(io.BytesIO(b"\x00\x01 not,a\ncsv"))
    assert clean_df is None and not rep.ok


def test_clean_raw_matches_bundled():
    c = clean()
    assert len(c) == 404717 and c["name"].nunique() == 31
    assert c.duplicated(["name", "timestamp"]).sum() == 0


# ---------------- methodology reproduction
def test_split_matches_notebook():
    data = build_training_table(clean())
    tr, va, te, t1, t2 = split_chronological(data)
    assert len(te) == 59668                                   # notebook / model_comparison_test.csv
    assert t2 == pd.Timestamp("2019-11-12 14:08:25")          # dashboard TEST_START
    assert tr.timestamp.max() < t1 <= va.timestamp.min() and va.timestamp.max() < t2 <= te.timestamp.min()


def test_no_feature_equals_current_target_inputs():
    data = build_training_table(clean())
    for f in C.FEATURES:
        assert not np.allclose(data[f].fillna(0), data["growth_rate"].fillna(0))
        assert not np.allclose(data[f].fillna(0), data["popularity_change"].fillna(0))


def test_features_use_only_past_data():
    """Changing FUTURE readings must not change features computed at an earlier row."""
    rng = np.random.RandomState(0)
    h = np.cumsum(rng.randint(-3, 4, 80)) + 1000.0
    base = compute_features(h)
    h2 = h.copy(); h2[50:] += 999
    alt = compute_features(h2)
    pd.testing.assert_frame_equal(base.iloc[:50], alt.iloc[:50])


def test_inference_features_equal_training_features():
    """compute_features at row i-1 == notebook training features at row i (one stock, steady state)."""
    c = clean(); g = c[c.name == "AAAU"].sort_values("timestamp").reset_index(drop=True)
    f = compute_features(g.users_holding.values); f["timestamp"] = g.timestamp
    f[C.FEATURES] = f[C.FEATURES].shift(1)
    d = build_training_table(c[c.name == "AAAU"]).merge(f, on="timestamp", suffixes=("", "_inf"))
    assert len(d) > 10000
    for k in C.FEATURES:
        assert np.nanmax(np.abs(d[k] - d[k + "_inf"])) < 1e-9, k


# ---------------- model
def test_model_compatibility_checks():
    assert validate_model_artifact(art()).ok
    bad = dict(art()); bad["features"] = C.FEATURES[:5]
    rep = validate_model_artifact(bad)
    assert not rep.ok
    swapped = dict(art()); swapped["features"] = C.FEATURES[::-1]
    assert "ORDER" in validate_model_artifact(swapped).errors[0]
    assert not validate_model_artifact({"model": 1}).ok
    assert not validate_model_artifact(dict(art(), threshold=1.7)).ok


def test_incompatible_input_is_refused():
    X = pd.DataFrame({"prev_users_holding": [1.0]})
    try:
        predict_proba_checked(art(), X)
        assert False, "should have raised"
    except ValueError as e:
        assert "missing feature" in str(e)


def test_portable_model_matches_saved_scores():
    """Probabilities must equal those the notebook saved (dashboard_scores.csv). A tiny number of CIG rows differ
    because of rolling-std numerics - see FINDINGS.md. Skipped if the reference file is not present."""
    ref_path = Path(os.environ.get("REF_SCORES", "/home/claude/work/outputs/dashboard_scores.csv"))
    if not ref_path.exists():
        return
    ref = pd.read_csv(ref_path, parse_dates=["timestamp"]).dropna(subset=["probability"])
    mine = pd.concat([score_series(g, art()) for _, g in clean().groupby("name")])
    m = ref.merge(mine[["name", "timestamp", "probability"]], on=["name", "timestamp"], suffixes=("_r", "_n"))
    bad = m[(m.probability_r - m.probability_n).abs() > 1e-4]
    assert len(m) > 390000 and len(bad) < 100 and set(bad.name) <= {"CIG"}


def test_numpy_export_equals_sklearn_for_all_model_types():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(1)
    X = pd.DataFrame(rng.randn(500, 7), columns=C.FEATURES); y = (X.iloc[:, 0] + rng.randn(500) > 0.8).astype(int)
    models = [Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression())]).fit(X, y),
              RandomForestClassifier(20, max_depth=5, random_state=0).fit(X, y),
              GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=0).fit(X, y)]
    for m in models:
        pm = PortableModel(export_model_to_dict(m, C.FEATURES, 0.5, "t"))
        assert np.abs(m.predict_proba(X)[:, 1] - pm.predict_proba(X)[:, 1]).max() < 1e-6, type(m).__name__


def test_label_mapping_and_predict_from_counts():
    res = predict_from_counts([1200 - 3 * i for i in range(20)], art())
    assert res["label"] in C.LABEL_MAP.values() and 0 <= res["probability"] <= 1
    assert abs(sum(res["class_probabilities"].values()) - 1) < 1e-9
    assert (res["label"] == C.HIGH_LABEL) == (res["probability"] >= art()["threshold"])
    for bad in ([1, 2, 3], [-1] * 20, [0.0] * 20):
        try:
            predict_from_counts(bad, art()); assert False
        except ValueError:
            pass


def test_evaluation_matches_notebook_results():
    res = evaluate_model(art(), clean())
    nb = pd.read_csv(C.ARTIFACTS_DIR / "model_comparison_test.csv", index_col=0).loc["Gradient Boosting"]
    for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
        assert abs(res["metrics"][k] - nb[k]) < 0.002, (k, res["metrics"][k], nb[k])


def test_retraining_reproduces_threshold():
    if os.environ.get("RUN_SLOW") != "1":
        return
    from pipelines.training import train_final_model
    new = train_final_model(clean(), out_dir=tempfile.mkdtemp())
    assert abs(new["threshold"] - art()["threshold"]) < 1e-9


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa
                fails += 1; print("FAIL", name, "->", repr(e)[:200])
    sys.exit(1 if fails else 0)
