# Explainable AI-Based Financial Transaction Fraud Detection System - Project Report Skeleton
> Sections marked **[FILL FROM YOUR RUN]** need real numbers from `models/metrics.json` after you train. Do not invent results.

## Abstract
Digital payments are growing and fraud is rare but costly. This project builds a web platform that scores financial transactions with a machine-learning model trained on the public PaySim dataset, classifies risk as Low/Medium/High, and explains each decision with SHAP so analysts can trust and audit it. Results: **[FILL FROM YOUR RUN]**.

## 1. Problem statement
Fraud is extremely imbalanced and black-box models are hard to trust. We need accurate, explainable, persistent fraud screening.

## 2. Objectives
Train and compare three classifiers; tune the threshold; explain predictions with SHAP; deliver an authenticated web app with history, analytics and monitoring.

## 3. Scope
In scope: PaySim-style TRANSFER/CASH_OUT transactions, single-transaction scoring, per-user storage. Out of scope: real bank integration, streaming, confirmed-fraud feedback loop.

## 4. Literature review outline
Rule-based systems; supervised learning for fraud (LR, RF, gradient boosting); imbalance techniques (class weights, SMOTE, undersampling); explainability (LIME, SHAP); concept drift. **[Add 8-10 real papers you read.]**

## 5. Existing vs proposed system
Existing: static rules and opaque scores. Proposed: learned model + tuned threshold + SHAP + audit history.

## 6. Software requirements (SRS summary)
Functional: register/login, analyze, risk + explanation, history search/filter/paginate, analytics, suspicious monitor, profile. Non-functional: responsive, password hashing, per-user isolation, reduced-motion support, model loaded once. Software: Python 3.10+, Flask, scikit-learn, XGBoost, SHAP, SQLAlchemy, Bootstrap, Chart.js. See `DIAGRAMS.md` for architecture, DFD, use case, ER and sequence diagrams.

## 7. Methodology
Dataset selection (PaySim: public, large, interpretable columns) -> EDA (`ml/eda.py`) -> preprocessing and leakage-safe feature engineering (`ml/features.py`: only pre-transaction information; post-transaction balances, `isFlaggedFraud` and identifiers excluded) -> stratified split -> tuning (RandomizedSearchCV, PR-AUC) -> model selection (validation PR-AUC) -> threshold tuning (validation) -> test evaluation -> SHAP -> export -> serve.

## 8. Algorithms
Logistic Regression (linear, interpretable baseline); Random Forest (bagged trees, handles non-linearity); XGBoost (gradient-boosted trees, strong on tabular data); SHAP (Shapley-value feature attribution; TreeExplainer/LinearExplainer).

## 9. Model evaluation results
**[FILL FROM YOUR RUN]** table: precision, recall, F1, ROC-AUC, PR-AUC, accuracy, confusion matrix for each model (copy from `metrics.json`), plus plots from `static/img/ml/`.

## 10. Testing
`tests/` covers features, risk levels, model-refusal safety, auth, validation, authorization isolation, history filters, stats, 503 on missing model, and a training smoke test. **[Record your actual pytest output here.]**

## 11. Limitations
Simulated data; PaySim fraud patterns may be easier than real ones; only two transaction types; no drift monitoring; threshold fits one dataset; SQLite for local use only.

## 12. Future scope
Feedback loop for confirmed fraud, batch CSV upload, REST API keys, model monitoring and retraining, IEEE-CIS dataset, graph features.

## 13. Conclusion
**[Write after results.]**

## Presentation outline (12-15 slides)
Title; problem; objectives; dataset + imbalance; architecture; preprocessing/features; models; metrics (why not accuracy); threshold trade-off; SHAP; live demo; testing; limitations/future; conclusion; Q&A.
