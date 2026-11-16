# Three-member division (work in parallel on GitHub, one branch per member)

| | Member 1 - AI/ML | Member 2 - Backend | Member 3 - Frontend |
|---|---|---|---|
| Owns | `ml/`, `models/`, `static/img/ml/` | `backend/`, `database/`, `tests/test_auth.py`, `tests/test_transactions.py`, `deployment/` | `templates/`, `static/css`, `static/js` |
| Tasks | Download PaySim; run EDA; run training; read metrics; explain model choice, threshold trade-off and SHAP; write the ML report chapters | API routes, validation, auth, DB models, predictor integration, error handling, API tests, Render deployment | Design system, landing/auth pages, dashboard, analyze form, result page with gauge + SHAP chart, history, analytics, responsiveness, accessibility |
| Tech | pandas, scikit-learn, XGBoost, imbalanced-learn, SHAP, joblib | Flask, SQLAlchemy, Flask-Login, pytest, gunicorn | HTML, CSS, JS, Bootstrap 5, Chart.js |
| Deliverables | trained artifacts + `metrics.json` + plots + ML chapter | working tested API + deployed URL + SRS/ER/DFD docs | polished UI + screenshots + UI chapter + demo script |
| Depends on | nothing (starts first) | M1's artifacts (`model_config.json`, `fraud_pipeline.joblib`) for real predictions | M2's API contract (see routes) |
| Integration | Commit `models/*` to Git; tell M2 | Run `pytest`; verify `/api/model/status` is `available:true` | Check each page against live API |

Workflow: `git checkout -b member1-ml` (etc.), small commits, pull requests into `main`, review each other's PRs.
Final week: everyone rehearses the demo; each member must be able to answer viva questions about ALL layers.
