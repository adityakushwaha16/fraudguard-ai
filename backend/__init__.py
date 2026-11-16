"""Flask application factory."""
from pathlib import Path
from flask import Flask, jsonify, redirect, request, url_for
from .config import Config, BASE_DIR
from .extensions import db, login_manager


def create_app(config_class=Config):
    app = Flask(__name__,
                template_folder=str(BASE_DIR / "templates"),
                static_folder=str(BASE_DIR / "static"))
    app.config.from_object(config_class)
    (BASE_DIR / "database").mkdir(exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)

    from .models import User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    @login_manager.unauthorized_handler
    def unauthorized():
        if request.path.startswith("/api/"):
            return jsonify({"error": "Authentication required"}), 401
        return redirect(url_for("pages.login_page"))

    from .routes import auth, pages, transactions
    from .services.predictor import Predictor
    app.extensions["predictor"] = Predictor(app.config["MODEL_DIR"])
    for m in (auth, pages, transactions):
        app.register_blueprint(m.bp)

    with app.app_context():
        db.create_all()

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    return app
