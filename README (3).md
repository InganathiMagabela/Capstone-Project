# Retail Investor Financial-Risk System (Capstone)

An educational **decision-support** application. It is not financial advice and does not predict investor losses.

## 1-2. Purpose and research problem
Retail platforms (e.g. Robinhood, EasyEquities) make investing accessible, but many retail investors lack risk awareness. This
project asks whether machine learning can give an early, understandable warning of **financial-risk behaviour**. Using the Robinhood
popularity data (number of users holding each stock over time), **risk = 1 when the number of holders decreases at the next
observation** (a proxy: the data hold no prices, trades or investor identities).

## 3. What the application does
Validates uploaded data -> cleans it -> engineers 7 features -> checks the model is compatible -> predicts the probability of a
decrease at the next observation -> shows the result (`Low risk` / `High risk`), a back-test, a watchlist, model performance,
feature importance and plain-language explanations. Diversification, drawdown and portfolio measures are intentionally absent
because the data cannot support them (see `FINDINGS.md`).

## 4. Structure
```
main.py              master script (pipeline checks, optional retraining, launches the dashboard)
dashboard.py         Streamlit interface
config/config.py     all constants from the notebook (features, split, labels, thresholds)
pipelines/           preprocessing, feature_engineering, training, evaluation, prediction, risk_analysis
utils/               validation, helpers, portable_model (NumPy copy of the model)
data/raw/            put robin_dataset.csv here (optional)   data/processed/clean_holdings.csv  (bundled)
models/              risk_model.joblib, model_export.json
artifacts/           notebook tables and figures (+ live_metrics.json written by main.py)
notebooks/           capstone.ipynb (original, unchanged)      tests/   test suite     FINDINGS.md  methodology review
```

## 5-6. Installation and Python version
Python 3.12 (the notebook used 3.12.4, scikit-learn 1.4.2).
```bash
pip install -r requirements.txt
python main.py
```
`python main.py --no-dashboard` runs the checks only; `--retrain` retrains (about 30 s; `--tune` repeats the search, slow).

## 7. Dataset requirements
CSV in long format, one row per stock per reading: `name` (ticker), `timestamp` (date-time), `users_holding` (whole number >= 0).
Rows named `robin_dataset` and duplicate (name, timestamp) rows are removed (as in the notebook). A stock needs at least 16
readings to be scored. Errors are shown in plain language, e.g. *"The dataset is missing the required column(s): users_holding"*.

## 8. Model requirements
A joblib file containing a dict `{"model", "features", "threshold", "model_name"}`; the model must have `predict_proba`, exactly the
7 features in `config.FEATURES` in that order, and a threshold between 0 and 1. Incompatible models are refused and no prediction
is produced. **Only upload joblib files you created: they are pickles and can run code when opened.**

## 9-10. Running and uploading
`python main.py` opens the dashboard (http://localhost:8501). In the sidebar you can optionally upload a data CSV and a model; with
no upload the bundled Capstone data and model are used.

## 11. How the risk analysis works
For each stock, in time order: popularity change, growth rate, 7-reading average, 7-reading momentum, current holders,
number of falls in the last 7 readings, and 7-reading volatility of the growth rate - all from observations up to now, never the
future. The model outputs a probability; `High risk` if it is >= the decision threshold (0.5678) chosen on validation data.

## 12. How the model works
Gradient Boosting (100 trees, depth 2, learning rate 0.1, balanced sample weights), trained on 70% of the timeline, threshold
tuned on the next 15%, refit on both, tested once on the final 15% (from 2019-11-12). Random seed 42.
Test results (Capstone test period; recalculated by `main.py`): accuracy 0.856, precision 0.271, recall 0.778, F1 0.402,
ROC-AUC 0.896, PR-AUC 0.635. **Excluding the data-feed artifact CIG, PR-AUC is 0.264.**

## 13. How to interpret results
Probability = chance that the *next reading* is a fall in holders. Falls are rare (~6% of test readings), so precision is modest and
many alerts are false alarms. Feature importance shows what the model uses, not what causes behaviour. `High risk` never means an
investor lost money.

## 14. Limitations
US data, 31 stocks, 2018-2020, aggregate (not individual) behaviour; EasyEquities / South African generalisation untested; CIG
artifact inflates headline scores; irregular reading intervals; one chronological test period; history does not guarantee the future.

## 15. Adding another model
Train it on `config.FEATURES` (same order), save `joblib.dump({"model": m, "features": config.FEATURES, "threshold": t,
"model_name": "..."}, "my_model.joblib")`, upload it in the sidebar or replace `models/risk_model.joblib`. For a new type
extend `utils/portable_model.py`.

## Tests
`python tests/test_pipeline.py` (set `RUN_SLOW=1` to include retraining) and `python tests/smoke_dashboard_stub.py`.
