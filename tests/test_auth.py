from .conftest import register


def test_register_login_logout_profile(client):
    assert register(client).status_code == 201
    assert client.get("/api/auth/profile").json["email"] == "a@example.com"
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/profile").status_code == 401
    assert client.post("/api/auth/login", json={"email": "a@example.com", "password": "password123"}).status_code == 200


def test_duplicate_and_validation(client):
    register(client)
    assert register(client).status_code == 409
    assert client.post("/api/auth/register", json={"full_name": "x", "email": "bad", "password": "password123"}).status_code == 400
    assert client.post("/api/auth/register", json={"full_name": "x", "email": "b@x.com", "password": "short"}).status_code == 400
    assert client.post("/api/auth/register", json={}).status_code == 400


def test_wrong_password(client):
    register(client); client.post("/api/auth/logout")
    assert client.post("/api/auth/login", json={"email": "a@example.com", "password": "nope"}).status_code == 401


def test_password_is_hashed(app, client):
    from backend.models import User
    register(client)
    with app.app_context():
        assert "password123" not in User.query.first().password_hash


def test_pages_redirect_when_logged_out(client):
    r = client.get("/dashboard"); assert r.status_code == 302 and "/login" in r.headers["Location"]
