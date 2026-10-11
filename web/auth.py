import re
import sqlite3
from urllib.parse import urlparse

from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for
from flask_login import UserMixin, current_user, login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from . import login_manager
from .db import get_db

bp = Blueprint("auth", __name__)
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class User(UserMixin):
    def __init__(self, row):
        self.id = row["id"]
        self.name = row["name"]
        self.email = row["email"]
        self.role = row["role"]

    @property
    def is_admin(self):
        return self.role == "admin"


@login_manager.user_loader
def load_user(user_id):
    row = get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return User(row) if row else None


@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith("/api/"):
        return jsonify(error="Sign in required."), 401
    flash("Sign in to continue.", "info")
    return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))


def safe_next(target):
    """Allow only same-site relative paths, so login cannot redirect off-site."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return None
    parsed = urlparse(target)
    return target if not parsed.scheme and not parsed.netloc else None


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("views.dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        row = get_db().execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if row and check_password_hash(row["pw_hash"], password):
            login_user(User(row), remember=bool(request.form.get("remember")))
            return redirect(safe_next(request.args.get("next")) or url_for("views.dashboard"))
        flash("The email or password is incorrect. Check both and try again.", "error")
    return render_template("login.html")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("views.dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        error = None
        if not name:
            error = "Enter your name."
        elif not EMAIL_RE.match(email):
            error = "Enter a valid email address."
        elif len(password) < 8:
            error = "Use a password with at least 8 characters."
        elif password != confirm:
            error = "The two passwords do not match."
        if error:
            flash(error, "error")
        else:
            conn = get_db()
            admin_email = current_app.config.get("ADMIN_EMAIL")
            is_admin = (email == admin_email) if admin_email else conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
            try:
                cur = conn.execute(
                    "INSERT INTO users (name, email, pw_hash, role) VALUES (?, ?, ?, ?)",
                    (name, email, generate_password_hash(password), "admin" if is_admin else "planner"),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                flash("An account with this email already exists. Sign in instead.", "error")
            else:
                row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
                login_user(User(row))
                return redirect(url_for("views.dashboard"))
    return render_template("register.html")


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You are signed out.", "info")
    return redirect(url_for("auth.login"))
