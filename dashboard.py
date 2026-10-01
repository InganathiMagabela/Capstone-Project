"""Retail Investor Financial-Risk Dashboard (Streamlit).   Start with:  python main.py

For a stock in the Robinhood popularity data, the model estimates the probability that the number of users
holding it will DECREASE at the next observation. This is a PROXY for financial risk - the data contain no
prices, trades, returns or investor identities. Decision-support / educational tool, not financial advice.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import config as C
from pipelines import prediction as pred
from pipelines import risk_analysis as ra
from pipelines.evaluation import evaluate_model
from pipelines.preprocessing import find_default_data, load_and_validate
from utils.validation import validate_model_artifact

TEST_START = pd.Timestamp("2019-11-12 14:08:25")   # verified to equal the notebook's test start


# ------------------------------------------------------------------ cached loaders
@st.cache_resource
def get_default_model():
    return pred.load_model()


@st.cache_data
def get_default_data():
    return load_and_validate(find_default_data())


@st.cache_data
def get_uploaded_data(raw_bytes):
    return load_and_validate(raw_bytes)


@st.cache_data
def get_quality(clean):
    return ra.quality_flags(clean)


@st.cache_data
def get_watchlist(_art, clean):
    return ra.latest_scores(clean, _art)


@st.cache_data
def get_live_eval(_art, clean):
    return evaluate_model(_art, clean)


def show_validation(rep, what):
    for e in rep.errors:
        st.error(f"ERROR in {what}: {e}")
    for w in rep.warnings:
        st.warning(w)


# ------------------------------------------------------------------ sidebar: DATA
def sidebar_inputs():
    st.sidebar.title("RETAIL INVESTMENT RISK SYSTEM")
    st.sidebar.subheader("DATA")
    up_data = st.sidebar.file_uploader("1. Holder-count dataset (CSV, optional)", type=["csv"],
                                       help="Long format with columns: name, timestamp, users_holding. "
                                            "Leave empty to use the Capstone data.")
    up_model = st.sidebar.file_uploader("2. Saved model (.joblib, optional)", type=["joblib", "pkl"],
                                        help="Dictionary with keys model, features, threshold. Only upload files "
                                             "you created yourself: joblib files can execute code when opened.")
    # --- data
    if up_data is not None:
        clean, rep = get_uploaded_data(up_data.getvalue())
        source = f"uploaded file '{up_data.name}'"
    else:
        path = find_default_data()
        if path is None:
            st.error(f"No data found. Upload a CSV or place robin_dataset.csv in {C.DATA_RAW}.")
            st.stop()
        clean, rep = get_default_data()
        source = f"bundled data ({Path(path).name})"
    show_validation(rep, "the dataset")
    if clean is None:
        st.stop()
    # --- model
    if up_model is not None:
        try:
            art, mrep = pred.load_model_from_bytes(up_model.getvalue())
            art["source"] = "uploaded"
        except Exception as exc:
            st.error(f"ERROR: the uploaded model could not be loaded. Details: {exc}")
            st.stop()
        if not mrep.ok:
            st.error("The uploaded model is INCOMPATIBLE, so no predictions are produced:")
            for e in mrep.errors:
                st.write("- " + e)
            st.stop()
        model_msg = f"Uploaded model {mrep.info['model_type']} - compatible"
    else:
        try:
            art, model_msg = get_default_model()
        except Exception as exc:
            st.error(f"Could not load the bundled model. Run `python main.py --retrain`. Details: {exc}")
            st.stop()
    mrep = validate_model_artifact(art)
    st.sidebar.success(f"Data: {rep.info['stocks']} stocks, {rep.info['rows']:,} rows")
    st.sidebar.success(f"Model: {art.get('model_name', 'unknown')} - {len(art['features'])} features "
                       f"{'compatible' if mrep.ok else 'INCOMPATIBLE'}")
    if art.get("source") == "portable":
        st.sidebar.info("Using the NumPy copy of the model (the pickle needs scikit-learn 1.4.2).")
    st.sidebar.caption(f"Data source: {source}")
    st.sidebar.subheader("ANALYSIS")
    page = st.sidebar.radio("Section", ["Risk Analysis", "Watchlist", "Try your own data", "Model Performance",
                                        "Explanations & Limitations"], label_visibility="collapsed")
    return clean, art, page, up_data is not None or up_model is not None


# ------------------------------------------------------------------ pages
def page_risk(clean, art):
    quality = get_quality(clean)
    stocks = sorted(clean["name"].unique())
    c0, c1, c2 = st.columns([2, 2, 3])
    stock = c0.selectbox("Stock", stocks)
    window = c1.slider("Observations to display", 100, 3000, 500, step=100)
    oos = c2.checkbox("Back-test on the out-of-sample test period only", value=True,
                      help=f"The model was trained on data before {TEST_START:%d %b %Y}.")
    scored = pred.score_series(clean[clean["name"] == stock], art)
    last = scored.iloc[-1]

    st.header("Risk Assessment")
    flag = quality.loc[stock]
    if flag["data_quality_flag"]:
        st.warning(f"Data-quality warning for {stock}: {flag['flag_reason']}. Predictions for this stock are not reliable.")

    st.subheader("Factual output")
    a, b, c, d = st.columns(4)
    a.metric("Latest holders", f"{int(last['users_holding']):,}")
    if pd.isna(last["probability"]):
        b.metric("Probability of a decrease (next observation)", "n/a")
        c.metric("Risk classification", "Insufficient history")
    else:
        b.metric("Probability of a decrease (next observation)", f"{last['probability']:.1%}")
        c.metric("Risk classification", C.HIGH_LABEL if last["alert"] else C.LOW_LABEL)
    d.metric("Last reading", f"{last['timestamp']:%Y-%m-%d %H:%M}")
    if not pd.isna(last["probability"]):
        st.write("Class probabilities: " + " | ".join(
            f"{C.LOW_LABEL} {1 - last['probability']:.0%}" if i == 0 else f"{C.HIGH_LABEL} {last['probability']:.0%}"
            for i in (0, 1)) + f"   (decision threshold {art['threshold']:.3f})")
    desc = ra.stock_descriptors(scored, window)
    e, f, g = st.columns(3)
    e.metric("Readings that were a fall (displayed window)", f"{desc['share_of_readings_that_fell']:.1%}")
    f.metric("Net change in holders (window)", f"{desc['net_change_over_window']:+,.0f}")
    g.metric("Latest vs peak in window", f"{desc['peak_to_latest_decline']:.1%}")

    st.subheader("Educational interpretation")
    if pd.isna(last["probability"]):
        st.info("There are not enough consecutive readings to compute every feature for this stock.")
    elif last["alert"]:
        st.info(f"The model's estimate is at or above its decision threshold, so it labels the next reading as "
                f"'{C.HIGH_LABEL}': a fall in the number of holders is more likely than usual. This is a statistical "
                "pattern in past data. It does not mean investors will lose money, and it can be wrong.")
    else:
        st.info(f"The model's estimate is below its decision threshold, so it labels the next reading as "
                f"'{C.LOW_LABEL}'. Falls are uncommon in the data{base_rate_text()}, so this is the usual situation. "
                "It is not a guarantee that holders will not fall.")

    view = scored.tail(window)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(view["timestamp"], view["users_holding"], lw=1.2)
    ax[0].set_ylabel("Users holding"); ax[0].set_title(f"{stock}: holders and predicted risk")
    ax[1].plot(view["timestamp"], view["probability"], lw=1, label="Predicted probability of a decrease")
    ax[1].axhline(art["threshold"], color="red", ls="--", lw=1, label="Decision threshold")
    dec = view[view["next_decrease"] == 1]
    ax[1].scatter(dec["timestamp"], [0.02] * len(dec), s=8, color="black", label="Actual decrease followed")
    ax[1].set_ylim(0, 1.02); ax[1].set_ylabel("Probability"); ax[1].legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    st.pyplot(fig); plt.close(fig)

    bt_data = scored[scored["timestamp"] >= TEST_START] if oos else scored
    bt = ra.backtest(bt_data)
    st.subheader("Back-test for this stock")
    if bt is None:
        st.info("No scored observations in the selected period for this stock.")
    else:
        w, x, y, z = st.columns(4)
        w.metric("Observations scored", f"{bt['observations']:,}")
        x.metric("Actual decreases", f"{bt['actual_decreases']:,}", f"base rate {bt['base_rate']:.1%}")
        y.metric("Alerts raised", f"{bt['alerts']:,}", f"{bt['alerts_correct']:,} correct")
        z.metric("Precision / recall", f"{bt['precision']:.2f} / {bt['recall']:.2f}")
        if not oos:
            st.caption("Includes observations the model was trained on, so it is optimistic.")


def test_base_rate():
    """Share of decreases in the notebook's test period = 1 - accuracy of the 'always Low risk' baseline."""
    p = C.ARTIFACTS_DIR / "model_comparison_test.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    return float(1 - df.loc["Baseline: always 0", "Accuracy"])


def base_rate_text():
    r = test_base_rate()
    return "" if r is None else f" (about {r:.0%} of test-period readings)"


def page_watchlist(clean, art):
    st.header("Latest risk signal for every stock")
    wl = get_watchlist(art, clean).join(get_quality(clean)[["data_quality_flag", "flag_reason"]])
    if st.checkbox("Hide stocks with data-quality flags", value=False):
        wl = wl[~wl["data_quality_flag"]]
    wl = wl.sort_values("probability_of_decrease", ascending=False)
    show = wl.copy()
    show["probability_of_decrease"] = show["probability_of_decrease"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.1%}")
    st.dataframe(show)
    st.download_button("Download watchlist (CSV)", wl.to_csv().encode("utf-8"), "watchlist.csv", "text/csv")
    st.caption("Each row scores the stock's most recent reading in the data file. The bundled data end in "
               "February 2020, so these are historical illustrations, not live signals.")


def page_own(art):
    st.header("Score your own holder-count history")
    st.write(f"Paste at least {C.MIN_HISTORY} consecutive holder counts for ONE stock, oldest first, separated by "
             "commas, spaces or new lines.")
    example = "1200, 1203, 1203, 1199, 1195, 1195, 1190, 1188, 1188, 1185, 1180, 1181, 1176, 1170, 1170, 1166, 1160"
    txt = st.text_area("Holder counts", example, height=120)
    if st.button("Predict"):
        try:
            vals = [float(v) for v in txt.replace(",", " ").split()]
            res = pred.predict_from_counts(vals, art)
        except ValueError as exc:
            st.error(f"ERROR: {exc}")
            return
        a, b = st.columns(2)
        a.metric("Probability of a decrease (next observation)", f"{res['probability']:.1%}")
        b.metric("Risk classification", res["label"])
        st.write("Features computed from the last observation:")
        st.dataframe(pd.DataFrame(res["features"], index=["value"]).T)


def page_performance(clean, art, custom):
    st.header("Model Performance")
    st.caption("All numbers below were calculated from the Capstone data and test period; none are typed in by hand.")
    p = C.ARTIFACTS_DIR / "model_comparison_test.csv"
    st.subheader("Notebook results: model comparison on the test period")
    if p.exists():
        st.dataframe(pd.read_csv(p, index_col=0))
    else:
        st.info("model_comparison_test.csv not found - not yet calculated.")

    st.subheader("Recalculated now with the loaded model")
    if custom:
        st.info("A custom dataset or model is loaded, so the Capstone test-period evaluation is not shown. "
                "Run `python main.py` for the saved evaluation.")
    else:
        res = get_live_eval(art, clean)
        m = res["metrics"]
        cols = st.columns(6)
        for col, k in zip(cols, ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]):
            col.metric(k.replace("_", "-"), f"{m[k]:.3f}")
        st.caption(f"Model: {res['model_name']} | threshold {res['threshold']:.4f} | test rows {res['test_rows']:,} "
                   f"from {res['test_start'][:10]} | share of decreases {res['test_positive_rate']:.1%} "
                   f"(accuracy is misleading here; always predicting 'Low risk' scores {res['baselines']['always_0']['Accuracy']:.1%}).")
        cm = res["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(3.6, 3))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center")
        ax.set_xticks([0, 1], [C.LOW_LABEL, C.HIGH_LABEL]); ax.set_yticks([0, 1], [C.LOW_LABEL, C.HIGH_LABEL])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test period)")
        st.pyplot(fig); plt.close(fig)
        ex = res["metrics_excluding_CIG"]
        st.warning(f"Without the stock CIG (a data-feed artifact) the test PR-AUC is {ex['PR_AUC']:.3f} and ROC-AUC "
                   f"{ex['ROC_AUC']:.3f}. The figures including CIG are inflated; the ones without are the more credible estimate.")

    st.subheader("Feature importance (permutation, from the notebook)")
    imp = C.ARTIFACTS_DIR / "permutation_importance.csv"
    if imp.exists():
        df = pd.read_csv(imp)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.barh(df["Feature"], df["Importance"], xerr=df["Std"]); ax.invert_yaxis()
        ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
        st.pyplot(fig); plt.close(fig)
        st.caption("This shows what the model relies on. It does not show what causes investors to reduce holdings.")
    for title, fn in [("ROC and precision-recall curves", "roc_pr_curves.png")]:
        if (C.ARTIFACTS_DIR / fn).exists():
            st.subheader(title); st.image(str(C.ARTIFACTS_DIR / fn))
    for title, fn in [("Sensitivity: with and without the CIG feed artifact", "cig_sensitivity.csv"),
                      ("Stock size / identity sanity check", "sanity_check_size_identity.csv")]:
        if (C.ARTIFACTS_DIR / fn).exists():
            st.subheader(title); st.dataframe(pd.read_csv(C.ARTIFACTS_DIR / fn, index_col=0))


def page_about():
    st.header("Explanations & Limitations")
    with st.expander("What does 'High risk' mean here?", expanded=True):
        st.write("Risk = 1 when the number of users holding a stock fell compared with the previous reading; "
                 "Risk = 0 when it stayed the same or rose. It is a **proxy** for financial risk. The dataset has no "
                 "prices, trades, returns or investor identities, so 'High risk' does not mean anyone lost money.")
    with st.expander("What is the probability?"):
        st.write("It is the model's estimate (0-100%) that the NEXT reading will be a fall in holders. Falls are rare"
                 f"{base_rate_text()}, so even a 'Low risk' label leaves some chance of a fall. "
                 "The label is High risk when the probability reaches the decision threshold chosen on validation data.")
    with st.expander("What are the inputs (features)?"):
        st.write("Seven numbers computed from a stock's own past readings: the last change in holders, the last "
                 "growth rate, the 7-reading average, the 7-reading momentum, the current holder count, how many of the "
                 "last 7 readings were falls, and how variable the growth rate was over the last 7 readings "
                 "(a measure of how unstable the holder count has been - the 'volatility' of popularity).")
    with st.expander("Why is diversification / drawdown / portfolio concentration not shown?"):
        st.write("The Capstone data are aggregate holder counts per stock. They contain no portfolios, prices or sectors, "
                 "so those measures cannot be calculated legitimately and are not displayed.")
    st.subheader("Limitations")
    st.write("- Data: aggregate holder counts for 31 Robinhood (US) stocks, May 2018 - Feb 2020; South African / EasyEquities generalisation is untested.\n"
             "- The model predicts stock-level outflow, not an individual investor's risk or behaviour.\n"
             "- One stock (CIG) has an implausible hourly pattern that inflates headline scores; flagged stocks are marked.\n"
             "- Falls are rare, so precision is modest (many alerts are false alarms; see Model Performance).\n"
             "- Readings are roughly hourly with irregular gaps, so 'the next observation' is not always the next hour.\n"
             "- Historical patterns do not guarantee future behaviour. This is an educational decision-support prototype, "
             "not financial advice.")


# ------------------------------------------------------------------ main
def main():
    st.set_page_config(page_title="Retail Investment Risk System", layout="wide")
    st.title("Retail Investment Risk Analysis")
    st.caption("Capstone: Using Machine Learning to Predict Financial Risk and Potential Losses Among Retail "
               "Investors on Digital Investment Platforms. Educational decision-support tool - not financial advice.")
    clean, art, page, custom = sidebar_inputs()
    if page == "Risk Analysis":
        page_risk(clean, art)
    elif page == "Watchlist":
        page_watchlist(clean, art)
    elif page == "Try your own data":
        page_own(art)
    elif page == "Model Performance":
        page_performance(clean, art, custom)
    else:
        page_about()


if __name__ == "__main__":
    main()
