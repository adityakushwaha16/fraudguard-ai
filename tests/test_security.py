"""Cross-user isolation. User A's data must be invisible to user B on EVERY endpoint,
and 'belongs to someone else' must be indistinguishable from 'does not exist'."""
import ast
from pathlib import Path
from flask import has_app_context
from .conftest import register, DRAIN, NORMAL

ROOT = Path(__file__).resolve().parent.parent


def _make(client, payload=DRAIN):
    r = client.post("/api/transactions/predict", json=payload)
    assert r.status_code == 201
    return r.json["id"]


# ---------- harness guard ----------
def test_harness_has_no_ambient_app_context(client):
    """Regression guard: an open app context would share flask.g (and the cached login user)
    between clients and silently invalidate every cross-user test."""
    assert not has_app_context()


def test_each_client_is_its_own_user(two_users):
    a, b = two_users
    assert a.get("/api/auth/profile").json["email"] == "a@x.com"
    assert b.get("/api/auth/profile").json["email"] == "b@x.com"


def test_transaction_is_stored_under_the_creating_user(app, two_users):
    from backend.models import Transaction, User
    a, b = two_users
    tid_a, tid_b = _make(a), _make(b, NORMAL)
    with app.app_context():
        owner = {t.id: t.user.email for t in Transaction.query.all()}
    assert owner == {tid_a: "a@x.com", tid_b: "b@x.com"}


# ---------- detail ----------
def test_detail_other_users_transaction_is_404(two_users):
    a, b = two_users
    tid = _make(a)
    assert a.get(f"/api/transactions/{tid}").status_code == 200
    assert b.get(f"/api/transactions/{tid}").status_code == 404


def test_other_user_and_nonexistent_are_indistinguishable(two_users):
    a, b = two_users
    tid = _make(a)
    other = b.get(f"/api/transactions/{tid}")
    missing = b.get("/api/transactions/999999")
    assert other.status_code == missing.status_code == 404
    assert other.json == missing.json
    assert other.get_data() == missing.get_data()


def test_unauthenticated_detail_is_401(app, two_users):
    a, _ = two_users
    tid = _make(a)
    assert app.test_client().get(f"/api/transactions/{tid}").status_code == 401


def test_result_page_is_scoped_too(two_users):
    a, b = two_users
    tid = _make(a)
    assert a.get(f"/transactions/{tid}").status_code == 200
    other, missing = b.get(f"/transactions/{tid}"), b.get("/transactions/999999")
    assert other.status_code == missing.status_code == 404
    assert other.get_data() == missing.get_data()


# ---------- history / search / filters ----------
def test_history_is_scoped(two_users):
    a, b = two_users
    tid = _make(a); _make(a, NORMAL)
    assert a.get("/api/transactions").json["total"] == 2
    h = b.get("/api/transactions").json
    assert h["total"] == 0 and h["items"] == []
    _make(b)                                           # B now has exactly one of their own
    hb = b.get("/api/transactions").json
    assert hb["total"] == 1
    assert tid not in [i["id"] for i in hb["items"]]
    assert a.get("/api/transactions").json["total"] == 2   # B's activity does not leak into A


def test_search_and_filters_cannot_reach_other_users_rows(two_users):
    a, b = two_users
    tid = _make(a)                                     # TRANSFER, Fraudulent, High
    for qs in (f"q={tid}", "q=TRANSFER", "label=Fraudulent", "risk=High", "type=TRANSFER",
               "per_page=100", "page=1&per_page=100&label=Fraudulent&risk=High"):
        assert b.get(f"/api/transactions?{qs}").json["total"] == 0, qs


def test_user_id_parameters_are_ignored(two_users):
    a, b = two_users
    _make(a)
    for qs in ("user_id=1", "user_id=2", "owner=a@x.com", "user=1"):
        assert b.get(f"/api/transactions?{qs}").json["total"] == 0, qs
        assert b.get(f"/api/suspicious?{qs}").json["total"] == 0, qs


# ---------- dashboard / analytics / suspicious ----------
def test_dashboard_analytics_suspicious_are_scoped(two_users):
    a, b = two_users
    _make(a); _make(a, NORMAL)
    sa = a.get("/api/dashboard/stats").json
    assert (sa["total"], sa["fraudulent"]) == (2, 1)
    sb = b.get("/api/dashboard/stats").json
    assert sb["total"] == 0 and sb["fraudulent"] == 0 and sb["recent_suspicious"] == []
    assert sb["risk_distribution"] == {"Low": 0, "Medium": 0, "High": 0}
    an = b.get("/api/analytics").json
    assert an["total"] == 0 and sum(an["probability_histogram"]) == 0
    assert an["avg_amount_fraud"] is None and an["avg_amount_legit"] is None
    assert all(v == {"legitimate": 0, "fraudulent": 0} for v in an["by_amount"].values())
    assert b.get("/api/suspicious").json == {"items": [], "total": 0}
    assert a.get("/api/suspicious").json["total"] == 1


def test_profile_transaction_count_is_scoped(two_users):
    a, b = two_users
    _make(a); _make(a, NORMAL); _make(b)
    assert a.get("/api/auth/profile").json["transaction_count"] == 2
    assert b.get("/api/auth/profile").json["transaction_count"] == 1


def test_every_data_endpoint_requires_login(app, two_users):
    a, _ = two_users
    tid = _make(a)
    anon = app.test_client()
    for url in (f"/api/transactions/{tid}", "/api/transactions", "/api/dashboard/stats",
                "/api/analytics", "/api/suspicious", "/api/auth/profile"):
        assert anon.get(url).status_code == 401, url
    assert anon.post("/api/transactions/predict", json=DRAIN).status_code == 401


# ---------- switching accounts in ONE browser ----------
def test_switching_accounts_in_the_same_browser_leaks_nothing(app):
    c = app.test_client()
    register(c, "a@x.com"); tid = _make(c)
    assert c.post("/api/auth/logout").status_code == 200
    assert c.get(f"/api/transactions/{tid}").status_code == 401
    register(c, "b@x.com")
    assert c.get(f"/api/transactions/{tid}").status_code == 404
    assert c.get("/api/transactions").json["total"] == 0
    c.post("/api/auth/logout")
    assert c.post("/api/auth/login", json={"email": "a@x.com", "password": "password123"}).status_code == 200
    assert c.get(f"/api/transactions/{tid}").status_code == 200


# ---------- static guard: no route may bypass the scoped helper ----------
FORBIDDEN_ATTRS = {("Transaction", "query"), ("Prediction", "query")}
FORBIDDEN_SESSION_CALLS = {"query", "get", "execute", "scalars", "scalar"}


def _violations(path):
    tree = ast.parse(path.read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and (node.value.id, node.attr) in FORBIDDEN_ATTRS:
            bad.append((path.name, node.lineno, f"{node.value.id}.{node.attr}"))
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_SESSION_CALLS \
                and isinstance(node.value, ast.Attribute) and node.value.attr == "session":
            bad.append((path.name, node.lineno, f"session.{node.attr}"))
    return bad


def test_routes_never_query_transactions_directly():
    """Only backend/services/scoping.py may touch Transaction.query; routes must use its helpers."""
    route_files = sorted((ROOT / "backend" / "routes").glob("*.py"))
    assert route_files, "no route files found"
    problems = [v for f in route_files for v in _violations(f)]
    assert problems == [], f"routes bypass the ownership-scoped helper: {problems}"
    scoping = (ROOT / "backend" / "services" / "scoping.py").read_text()
    assert "Transaction.user_id == current_user.id" in scoping


def test_static_guard_detects_a_violation(tmp_path):
    """The guard itself must catch a bypass (so it cannot pass vacuously)."""
    bad = tmp_path / "evil.py"
    bad.write_text("def f(i):\n    return Transaction.query.get(i)\n\ndef g(i):\n    return db.session.get(Transaction, i)\n")
    assert len(_violations(bad)) == 2
