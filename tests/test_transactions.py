from .conftest import register, DRAIN, NORMAL


def test_predict_and_history(client):
    register(client)
    r = client.post("/api/transactions/predict", json=DRAIN); assert r.status_code == 201
    assert r.json["label"] == "Fraudulent" and r.json["risk_level"] == "High"
    assert client.post("/api/transactions/predict", json=NORMAL).json["label"] == "Legitimate"
    h = client.get("/api/transactions").json; assert h["total"] == 2
    assert client.get("/api/transactions?label=Fraudulent").json["total"] == 1
    assert client.get("/api/transactions?risk=Low").json["total"] == 1
    assert client.get("/api/transactions?q=TRANSFER").json["total"] == 1
    assert client.get("/api/transactions?per_page=1").json["pages"] == 2


def test_invalid_inputs(client):
    register(client)
    bad = [{**DRAIN, "type": "PAYMENT"}, {**DRAIN, "amount": -5}, {**DRAIN, "amount": 0},
           {**DRAIN, "hour_of_day": 24}, {**DRAIN, "hour_of_day": -1}, {**DRAIN, "hour_of_day": 3.5},
           {k: v for k, v in DRAIN.items() if k != "hour_of_day"},
           {**DRAIN, "amount": "abc"}, {k: v for k, v in DRAIN.items() if k != "amount"}, {**DRAIN, "oldbalanceOrg": "nan"}]
    for b in bad:
        assert client.post("/api/transactions/predict", json=b).status_code == 400, b
    assert client.post("/api/transactions/predict", data="x").status_code == 400
    assert client.get("/api/transactions?date_from=garbage").status_code == 400


def test_requires_login(client):
    assert client.post("/api/transactions/predict", json=DRAIN).status_code == 401
    assert client.get("/api/dashboard/stats").status_code == 401


def test_users_cannot_see_each_others_data(app):
    a, b = app.test_client(), app.test_client()
    register(a, "a@x.com"); register(b, "b@x.com")
    tid = a.post("/api/transactions/predict", json=DRAIN).json["id"]
    assert a.get(f"/api/transactions/{tid}").status_code == 200
    assert b.get(f"/api/transactions/{tid}").status_code == 404
    assert b.get("/api/transactions").json["total"] == 0
    assert b.get("/api/dashboard/stats").json["total"] == 0


def test_stats_and_analytics(client):
    register(client)
    assert client.get("/api/dashboard/stats").json["total"] == 0
    client.post("/api/transactions/predict", json=DRAIN); client.post("/api/transactions/predict", json=NORMAL)
    s = client.get("/api/dashboard/stats").json
    assert (s["total"], s["fraudulent"], s["legitimate"], s["fraud_rate"]) == (2, 1, 1, 50.0)
    assert s["risk_distribution"]["High"] == 1 and len(s["recent_suspicious"]) == 1
    a = client.get("/api/analytics").json; assert a["total"] == 2 and sum(a["probability_histogram"]) == 2
    assert client.get("/api/suspicious").json["total"] == 1


def test_model_unavailable_returns_503(app, client):
    from backend.services.predictor import Predictor
    app.extensions["predictor"] = Predictor("/nonexistent")
    register(client)
    r = client.post("/api/transactions/predict", json=DRAIN)
    assert r.status_code == 503 and "model" in r.json["error"].lower()
    assert client.get("/api/model/status").json["available"] is False


def test_post_transaction_fields_are_ignored_not_required_not_stored(client):
    """Leakage guard at the API: balances AFTER the transaction must have no effect."""
    register(client)
    plain = client.post("/api/transactions/predict", json=DRAIN).json
    with_post = client.post("/api/transactions/predict",
                            json={**DRAIN, "newbalanceOrig": 123456, "newbalanceDest": 999}).json
    assert plain["label"] == with_post["label"]
    assert plain["fraud_probability"] == with_post["fraud_probability"]
    for key in ("newbalanceOrig", "newbalanceDest", "step"):
        assert key not in with_post
