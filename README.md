# FraudGuard AI - Explainable AI-Based Financial Transaction Fraud Detection

Flask + scikit-learn/XGBoost + SHAP + SQLAlchemy + Bootstrap 5 + Chart.js.
Dataset: **PaySim** (public, simulated mobile-money transactions). No real banking integration.
A prediction is an **estimate of fraud risk, not confirmed fraud**.

## IMPORTANT: the model is not included yet
No dataset was available when this code was generated, so **no model has been trained and no metrics exist**.
Until you train one, the website runs but refuses to predict (HTTP 503 "model not available").
It never produces made-up scores.

## Features and data-leakage policy
The model may only use information known **before** a transaction executes. Model features
(`ml/features.py`):

| Feature | Meaning |
|---|---|
| `type` | TRANSFER or CASH_OUT (one-hot encoded) |
| `amount` | transaction amount |
| `oldbalanceOrg`, `oldbalanceDest` | sender / receiver balance **before** the transaction |
| `hour_of_day` | PaySim `step % 24` |
| `amount_to_orig_balance` | amount / sender balance before (0 if balance is 0, capped at 10) |
| `drains_origin` | 1 if the amount would empty the sender account |
| `orig_balance_zero`, `dest_balance_zero` | 1 if that balance is 0 before the transaction |

**Excluded on purpose:** `newbalanceOrig`, `newbalanceDest` (outcomes of the transaction, so they leak the
answer), anything derived from them (`errorBalance*`, `orig_emptied`), `isFlaggedFraud` (the simulator's own
rule-based flag), `nameOrig`/`nameDest` (identifiers) and the absolute `step` (only its hour of day is kept).
The website form asks only for the five pre-transaction inputs (type, hour, amount, two "before" balances).
Training and the website call the same `engineer_features()`; the app refuses any saved model whose feature list
differs from the code (so models trained with the old feature set are rejected - just retrain).
Expect honest scores to be noticeably lower than models that peek at post-transaction balances; that is the
correct, deployable behaviour. Report whatever `metrics.json` says.

## Security: per-user data isolation
Every transaction read goes through `backend/services/scoping.py`, which adds `WHERE user_id = <logged-in user>`
to the SQL query. Detail, history (including search and filters), dashboard, analytics, suspicious-monitor and the
HTML result page all use it. A transaction that belongs to someone else returns the same 404 as one that does not
exist, so IDs cannot be probed. `tests/test_security.py` checks this with two logged-in users on every endpoint, and
a static test fails if any route queries `Transaction`/`Prediction` directly.
Note for test authors: never hold `app.app_context()` open while making test requests (see `tests/conftest.py`);
it makes all test clients share one logged-in user.

## Windows setup (VS Code)
1. Install Python 3.10-3.12 from python.org (tick "Add Python to PATH"), Git, and VS Code.
2. Open the project folder in VS Code, then open a terminal (PowerShell):
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # then edit SECRET_KEY in .env
```
3. Download PaySim from Kaggle ("PaySim synthetic financial datasets for fraud detection", ~470 MB CSV)
   and place the CSV in `datasets\`.
4. Explore + train (Member 1):
```powershell
python -m ml.eda   --data datasets\<paysim-file>.csv
python -m ml.train --data datasets\<paysim-file>.csv --sample 500000   # quick first run
python -m ml.train --data datasets\<paysim-file>.csv                   # full run (slow; hours possible)
```
   This writes `models\fraud_pipeline.joblib`, `model_config.json`, `metrics.json` and plots in `static\img\ml\`.
5. Run the website:
```powershell
python run.py          # open http://127.0.0.1:5000
```
6. Run tests: `python -m pytest -v`

The database (SQLite, `database\fraudguard.db`) is created automatically on first start.
If you ran an earlier version of this project, **delete `database\fraudguard.db` first**: the transactions table changed
(post-transaction columns removed, `hour_of_day` added).

## Project structure
| Path | Purpose |
|---|---|
| `ml/features.py` | Feature engineering used by BOTH training and inference |
| `ml/eda.py`, `ml/train.py` | EDA; tuning of LR / RF / XGBoost, threshold tuning, evaluation, SHAP, export |
| `ml/explain.py`, `ml/synthetic.py` | SHAP helpers; synthetic data for pipeline smoke tests only |
| `backend/` | Flask app: `models.py`, `routes/` (auth, transactions, pages), `services/predictor.py` |
| `templates/`, `static/` | Jinja pages, CSS design system, JS, ML plots |
| `models/` | Trained artifacts (created by `ml.train`) |
| `tests/` | pytest suite |
| `deployment/`, `Procfile` | Render config |
| `documentation/` | Report, diagrams, team roles, viva questions |

## How the model is chosen (no magic)
Rows are filtered to TRANSFER/CASH_OUT (the only types containing fraud in PaySim), de-duplicated, split
stratified 60/20/20, and reduced to the pre-transaction features above. Preprocessing lives **inside** the sklearn Pipeline (fitted on training folds only: no leakage).
Class imbalance is handled with class weights / `scale_pos_weight`. Three models are tuned with randomized search
(scoring = PR-AUC); the winner is picked by **validation PR-AUC**; the decision threshold is tuned on validation
(F-beta, default F1; `--beta 2` favours recall); final numbers are reported on the untouched test set.
Accuracy alone is misleading because ~99.9% of PaySim rows are legitimate.

## Deployment (Render + PostgreSQL)
1. Push to GitHub (commit `models/*.joblib` and `models/*.json` after training).
2. On Render: New > Blueprint (use `deployment/render.yaml`) or create a Web Service + PostgreSQL manually.
3. Set `SECRET_KEY`; `DATABASE_URL` comes from the Render database. Start command is in `Procfile`.
4. Verify current free-tier limits (sleep on idle, database expiry) on Render's pricing page before the demo.
For a live demo, a local run on a laptop is the most reliable.

## About Hatchable
Hatchable runs JavaScript functions and static pages; it cannot run Python/scikit-learn/SHAP. So the core
product is this Flask app. Hatchable can optionally host a public landing page that links to your Render URL.
All source here is plain files you own; nothing depends on Hatchable or Claude.
