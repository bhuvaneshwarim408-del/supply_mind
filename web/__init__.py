import os
import secrets

from flask import Flask, abort, jsonify, request, session
from flask_login import LoginManager

from . import db

login_manager = LoginManager()
login_manager.login_view = "auth.login"


def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        # Set SECRET_KEY in the environment for a stable key. Without it,
        # everyone is signed out whenever the server restarts.
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        DATABASE=os.path.join(app.instance_path, "sawtooth.db"),
        APP_NAME="SupplyMind",
        TAGLINE="AI for supply chain",
        CURRENCY="",
        # SUPPLYMIND_PUBLIC=1 lets anyone VIEW the dashboards without an account. Changing data still needs login.
        PUBLIC_VIEW=os.environ.get("SUPPLYMIND_PUBLIC") == "1",
        # If set, only this email becomes admin. If not set, the first account created is the admin (fine on a laptop,
        # NOT safe on a public site, so set it when deploying).
        ADMIN_EMAIL=(os.environ.get("SUPPLYMIND_ADMIN_EMAIL") or "").strip().lower(),
    )
    if config:
        app.config.update(config)
    os.makedirs(app.instance_path, exist_ok=True)

    if app.config["PUBLIC_VIEW"]:
        from werkzeug.middleware.proxy_fix import ProxyFix
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)   # correct client IP behind a hosting proxy

    db.init_app(app)
    login_manager.init_app(app)

    from .auth import bp as auth_bp
    from .views import bp as views_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(views_bp)

    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def csrf_protect():
        if request.method == "POST":
            sent = request.form.get("_csrf", "")
            expected = session.get("_csrf", "")
            if not expected or not secrets.compare_digest(sent, expected):
                abort(400, "Your session expired. Reload the page and try again.")

    from .services.real_backend import NotAnalysed

    @app.errorhandler(NotAnalysed)
    def not_analysed(err):
        msg = f"{err} has not been analysed on this server yet. Pick another SKU."
        return jsonify(error=msg), 404

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(429)
    def api_errors(err):
        if request.path.startswith("/api/"):
            return jsonify(error=err.description), err.code
        return err

    @app.context_processor
    def inject_globals():
        from .services import data_source

        return {
            "app_name": app.config["APP_NAME"],
            "tagline": app.config["TAGLINE"],
            "public_view": app.config["PUBLIC_VIEW"],
            "currency": app.config["CURRENCY"],
            "is_demo": data_source.IS_DEMO,
        }

    return app
