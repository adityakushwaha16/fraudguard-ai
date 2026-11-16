"""Auth API: register, login, logout, profile."""
import re
from flask import Blueprint, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from ..extensions import db
from ..models import User

bp = Blueprint("auth", __name__, url_prefix="/api/auth")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _err(msg, code=400):
    return jsonify({"error": msg}), code


@bp.post("/register")
def register():
    data = request.get_json(silent=True) or {}
    name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not name or not email or not password:
        return _err("full_name, email and password are required")
    if not EMAIL_RE.match(email):
        return _err("Invalid email address")
    if len(password) < 8:
        return _err("Password must be at least 8 characters")
    if User.query.filter_by(email=email).first():
        return _err("Email already registered", 409)
    user = User(full_name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return jsonify({"message": "Registered", "user": {"id": user.id, "email": user.email}}), 201


@bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    user = User.query.filter_by(email=(data.get("email") or "").strip().lower()).first()
    if not user or not user.check_password(data.get("password") or ""):
        return _err("Invalid email or password", 401)
    login_user(user)
    return jsonify({"message": "Logged in"})


@bp.post("/logout")
@login_required
def logout():
    logout_user()
    return jsonify({"message": "Logged out"})


@bp.get("/profile")
@login_required
def profile():
    return jsonify({"id": current_user.id, "full_name": current_user.full_name,
                    "email": current_user.email,
                    "member_since": current_user.created_at.isoformat(),
                    "transaction_count": current_user.transactions.count()})
