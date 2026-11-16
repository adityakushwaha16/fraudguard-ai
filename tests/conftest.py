import pytest
from backend import create_app
from backend.config import TestConfig
from backend.extensions import db


class FakePredictor:
    """TEST DOUBLE used only in automated tests (never in the real app)."""
    threshold = 0.5

    def status(self):
        return {"available": True, "model_name": "fake", "model_version": "t", "threshold": 0.5,
                "dataset": "test", "comparison": {}, "test_metrics": {}}

    def predict(self, tx):
        prob = 0.95 if tx["amount"] == tx["oldbalanceOrg"] else 0.02
        return {"fraud_probability": prob, "label": "Fraudulent" if prob >= .5 else "Legitimate",
                "risk_level": "High" if prob >= .5 else "Low", "threshold": .5, "model_version": "t",
                "explanation": {"summary": "test", "contributions": []}, "explanation_error": None}


@pytest.fixture()
def app():
    app = create_app(TestConfig)
    app.extensions["predictor"] = FakePredictor()
    with app.app_context():
        db.create_all()
    # IMPORTANT: do NOT keep an app context open while the test runs. Flask reuses an already-active
    # app context for requests, so flask.g (where Flask-Login caches the logged-in user) would be
    # shared between requests AND between test clients, making two "different" users look identical.
    # Production creates a fresh app context per request; the tests must do the same.
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def two_users(app):
    """Two independent browsers: user A and user B, both registered and logged in."""
    a, b = app.test_client(), app.test_client()
    assert register(a, "a@x.com").status_code == 201
    assert register(b, "b@x.com").status_code == 201
    return a, b


def register(client, email="a@example.com", pw="password123"):
    return client.post("/api/auth/register", json={"full_name": "A User", "email": email, "password": pw})


DRAIN = dict(type="TRANSFER", hour_of_day=5, amount=1000, oldbalanceOrg=1000, oldbalanceDest=0)
NORMAL = dict(type="CASH_OUT", hour_of_day=5, amount=100, oldbalanceOrg=5000, oldbalanceDest=10)
