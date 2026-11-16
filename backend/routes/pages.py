"""Server-rendered HTML pages. Data is loaded by JavaScript from the /api endpoints."""
from flask import Blueprint, abort, render_template, redirect, url_for
from flask_login import login_required, current_user

from ..services.scoping import get_owned_transaction

bp = Blueprint("pages", __name__)


@bp.get("/")
def landing():
    return render_template("landing.html")


@bp.get("/login")
def login_page():
    return redirect(url_for("pages.dashboard")) if current_user.is_authenticated else render_template("login.html")


@bp.get("/register")
def register_page():
    return redirect(url_for("pages.dashboard")) if current_user.is_authenticated else render_template("register.html")


def _page(name, template):
    @login_required
    def view():
        return render_template(template, page=name)
    view.__name__ = f"{name}_view"
    return view


for _name, _tpl, _path in [("dashboard", "dashboard.html", "/dashboard"), ("detect", "detect.html", "/detect"),
                           ("history", "history.html", "/history"), ("analytics", "analytics.html", "/analytics"),
                           ("suspicious", "suspicious.html", "/suspicious"), ("profile", "profile.html", "/profile")]:
    bp.add_url_rule(_path, endpoint=_name, view_func=_page(_name, _tpl))


@bp.get("/transactions/<int:tx_id>")
@login_required
def result(tx_id):
    if get_owned_transaction(tx_id) is None:   # identical 404 for missing and other-user IDs
        abort(404)
    return render_template("result.html", page="result", tx_id=tx_id)
