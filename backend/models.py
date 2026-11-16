"""Database models: User, Transaction, Prediction (one-to-one with Transaction)."""
from datetime import datetime, timezone
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from .extensions import db


def utcnow():
    return datetime.now(timezone.utc)


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)
    transactions = db.relationship("Transaction", backref="user", lazy="dynamic",
                                   cascade="all, delete-orphan")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Transaction(db.Model):
    """Only PRE-transaction inputs are stored (see ml/features.py leakage policy)."""
    __tablename__ = "transactions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    hour_of_day = db.Column(db.Integer, nullable=False, default=0)
    type = db.Column(db.String(20), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    oldbalance_org = db.Column(db.Float, nullable=False)
    oldbalance_dest = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, index=True)
    prediction = db.relationship("Prediction", backref="transaction", uselist=False,
                                 cascade="all, delete-orphan")


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey("transactions.id"),
                               unique=True, nullable=False)
    fraud_probability = db.Column(db.Float, nullable=False)
    label = db.Column(db.String(12), nullable=False)       # "Fraudulent" | "Legitimate"
    risk_level = db.Column(db.String(8), nullable=False)    # Low | Medium | High
    threshold_used = db.Column(db.Float, nullable=False)
    explanation_json = db.Column(db.Text)                   # SHAP contributions (JSON)
    model_version = db.Column(db.String(40))
    created_at = db.Column(db.DateTime, default=utcnow)
