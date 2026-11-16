"""Transaction, prediction, history, dashboard and analytics APIs (all per-user)."""
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required, current_user
from sqlalchemy import or_
from ..extensions import db
from ..models import Transaction, Prediction
from ..services.predictor import ModelNotAvailable
from ..services.scoping import get_owned_transaction, owned_transactions

bp = Blueprint("tx", __name__, url_prefix="/api")
TYPES = ("TRANSFER", "CASH_OUT")


def err(msg, code=400, **extra):
    return jsonify({"error": msg, **extra}), code


def validate_tx(data):
    """Returns (clean_dict, error_message). Only PRE-transaction fields are read;
    any other keys in the request (e.g. post-transaction balances) are ignored."""
    if not isinstance(data, dict):
        return None, "JSON body required"
    t = str(data.get("type", "")).upper()
    if t not in TYPES:
        return None, f"type must be one of {', '.join(TYPES)} (the model is trained only on these)"
    clean = {"type": t}
    if data.get("hour_of_day") in (None, ""):
        return None, "hour_of_day is required"
    try:
        h = float(data["hour_of_day"])
    except (TypeError, ValueError):
        return None, "hour_of_day must be an integer between 0 and 23"
    if h != int(h) or not 0 <= h <= 23:
        return None, "hour_of_day must be an integer between 0 and 23"
    clean["hour_of_day"] = int(h)
    for f in ("amount", "oldbalanceOrg", "oldbalanceDest"):
        if data.get(f) in (None, ""):
            return None, f"{f} is required"
        try:
            v = float(data[f])
        except (TypeError, ValueError):
            return None, f"{f} must be a number"
        if v != v or v in (float("inf"), float("-inf")) or v < 0:
            return None, f"{f} must be a finite number >= 0"
        clean[f] = v
    if clean["amount"] <= 0:
        return None, "amount must be greater than 0"
    return clean, None


def serialize(tx):
    p = tx.prediction
    d = {"id": tx.id, "created_at": tx.created_at.isoformat(), "hour_of_day": tx.hour_of_day,
         "type": tx.type, "amount": tx.amount, "oldbalanceOrg": tx.oldbalance_org,
         "oldbalanceDest": tx.oldbalance_dest}
    if p:
        d.update({"label": p.label, "fraud_probability": p.fraud_probability, "risk_level": p.risk_level,
                  "threshold": p.threshold_used, "model_version": p.model_version,
                  "explanation": json.loads(p.explanation_json) if p.explanation_json else None})
    return d


@bp.get("/model/status")
def model_status():
    return jsonify(current_app.extensions["predictor"].status())


@bp.post("/transactions/predict")
@login_required
def predict():
    clean, e = validate_tx(request.get_json(silent=True))
    if e:
        return err(e)
    predictor = current_app.extensions["predictor"]
    try:
        res = predictor.predict(clean)
    except ModelNotAvailable as exc:
        return err(str(exc), 503)
    tx = Transaction(user_id=current_user.id, hour_of_day=clean["hour_of_day"], type=clean["type"],
                     amount=clean["amount"], oldbalance_org=clean["oldbalanceOrg"],
                     oldbalance_dest=clean["oldbalanceDest"])
    pred = Prediction(transaction=tx, fraud_probability=res["fraud_probability"], label=res["label"],
                      risk_level=res["risk_level"], threshold_used=res["threshold"],
                      model_version=res["model_version"],
                      explanation_json=json.dumps(res["explanation"]) if res["explanation"] else None)
    db.session.add_all([tx, pred])
    db.session.commit()
    out = serialize(tx)
    out["explanation_error"] = res["explanation_error"]
    return jsonify(out), 201


@bp.get("/transactions")
@login_required
def history():
    q = owned_transactions().join(Prediction)
    s = (request.args.get("q") or "").strip()
    if s:
        conds = [Transaction.type.ilike(f"%{s}%")]
        if s.isdigit():
            conds.append(Transaction.id == int(s))
        q = q.filter(or_(*conds))
    if request.args.get("label") in ("Fraudulent", "Legitimate"):
        q = q.filter(Prediction.label == request.args["label"])
    if request.args.get("risk") in ("Low", "Medium", "High"):
        q = q.filter(Prediction.risk_level == request.args["risk"])
    if request.args.get("type") in TYPES:
        q = q.filter(Transaction.type == request.args["type"])
    for key, op in (("date_from", "ge"), ("date_to", "le")):
        v = request.args.get(key)
        if v:
            try:
                d = datetime.fromisoformat(v)
            except ValueError:
                return err(f"{key} must be YYYY-MM-DD")
            if op == "le":
                d += timedelta(days=1)
                q = q.filter(Transaction.created_at < d)
            else:
                q = q.filter(Transaction.created_at >= d)
    try:
        page = max(int(request.args.get("page", 1)), 1)
        per = min(max(int(request.args.get("per_page", 10)), 1), 100)
    except ValueError:
        return err("page and per_page must be integers")
    p = q.order_by(Transaction.created_at.desc()).paginate(page=page, per_page=per, error_out=False)
    items = []
    for t in p.items:
        d = serialize(t); d.pop("explanation", None); items.append(d)
    return jsonify({"items": items, "page": p.page, "pages": p.pages, "total": p.total})


@bp.get("/transactions/<int:tx_id>")
@login_required
def detail(tx_id):
    tx = get_owned_transaction(tx_id)   # ownership enforced in the SQL query
    if tx is None:                      # same 404 for 'does not exist' and 'belongs to someone else'
        return err("Transaction not found", 404)
    return jsonify(serialize(tx))


def _rows():
    return [(t, t.prediction) for t in owned_transactions().join(Prediction).all()]


@bp.get("/dashboard/stats")
@login_required
def stats():
    rows = _rows()
    total = len(rows)
    fraud = sum(1 for _, p in rows if p.label == "Fraudulent")
    risk = Counter(p.risk_level for _, p in rows)
    recent = sorted([(t, p) for t, p in rows if p.label == "Fraudulent" or p.risk_level in ("Medium", "High")],
                    key=lambda r: r[0].created_at, reverse=True)[:5]
    daily = defaultdict(lambda: [0, 0])
    for t, p in rows:
        daily[t.created_at.date().isoformat()][0] += 1
        daily[t.created_at.date().isoformat()][1] += p.label == "Fraudulent"
    days = sorted(daily)[-14:]
    return jsonify({"total": total, "fraudulent": fraud, "legitimate": total - fraud,
                    "fraud_rate": round(fraud / total * 100, 2) if total else 0.0,
                    "risk_distribution": {k: risk.get(k, 0) for k in ("Low", "Medium", "High")},
                    "recent_suspicious": [serialize(t) | {"explanation": None} for t, _ in recent],
                    "trend": {"dates": days, "total": [daily[d][0] for d in days],
                              "fraudulent": [daily[d][1] for d in days]}})


@bp.get("/analytics")
@login_required
def analytics():
    rows = _rows()
    buckets = [("<1K", 0, 1e3), ("1K-10K", 1e3, 1e4), ("10K-100K", 1e4, 1e5),
               ("100K-1M", 1e5, 1e6), (">1M", 1e6, float("inf"))]
    by_amount = {b[0]: {"legitimate": 0, "fraudulent": 0} for b in buckets}
    by_type = {t: {"legitimate": 0, "fraudulent": 0} for t in TYPES}
    prob_hist = [0] * 10
    for t, p in rows:
        key = "fraudulent" if p.label == "Fraudulent" else "legitimate"
        for name, lo, hi in buckets:
            if lo <= t.amount < hi:
                by_amount[name][key] += 1
        by_type[t.type][key] += 1
        prob_hist[min(int(p.fraud_probability * 10), 9)] += 1
    return jsonify({"total": len(rows), "by_amount": by_amount, "by_type": by_type,
                    "probability_histogram": prob_hist,
                    "avg_amount_fraud": _avg([t.amount for t, p in rows if p.label == "Fraudulent"]),
                    "avg_amount_legit": _avg([t.amount for t, p in rows if p.label == "Legitimate"])})


def _avg(v):
    return round(sum(v) / len(v), 2) if v else None


@bp.get("/suspicious")
@login_required
def suspicious():
    q = owned_transactions().join(Prediction).filter(
        or_(Prediction.label == "Fraudulent", Prediction.risk_level.in_(("Medium", "High"))))
    if request.args.get("risk") in ("Medium", "High"):
        q = q.filter(Prediction.risk_level == request.args["risk"])
    items = [serialize(t) | {"explanation": None}
             for t in q.order_by(Prediction.fraud_probability.desc()).limit(200).all()]
    return jsonify({"items": items, "total": len(items)})
