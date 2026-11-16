"""Application configuration. Values come from environment variables (.env)."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _db_url() -> str:
    url = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'database' / 'fraudguard.db'}")
    # Render/Heroku style URLs start with postgres:// which SQLAlchemy rejects
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-insecure-key")
    SQLALCHEMY_DATABASE_URI = _db_url()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MODEL_DIR = BASE_DIR / "models"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


class TestConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
