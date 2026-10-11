import sys
import types
from types import SimpleNamespace

import pytest

from web import create_app
from web.services import data_source, real_backend


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    from web import views
    views._hits.clear()


@pytest.fixture
def app(tmp_path):
    return create_app({"TESTING": True, "SECRET_KEY": "test", "DATABASE": str(tmp_path / "t.db")})


@pytest.fixture
def client(app):
    return app.test_client()


def token(client, path="/login"):
    client.get(path)
    with client.session_transaction() as s:
        return s["_csrf"]


def register(client, email="a@b.co", password="longenough1", name="Asha"):
    return client.post("/register", data={
        "_csrf": token(client, "/register"), "name": name, "email": email,
        "password": password, "confirm": password,
    })


# ---------- auth ----------
def test_pages_require_login(client):
    r = client.get("/")
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert client.get("/api/dashboard").status_code == 401


def test_post_without_csrf_rejected(client):
    assert client.post("/login", data={"email": "a@b.co", "password": "x"}).status_code == 400


def test_register_roles_and_duplicates(client):
    assert register(client).status_code == 302
    assert b"Admin" in client.get("/").data
    client.post("/logout", data={"_csrf": token(client, "/")})
    assert register(client, email="c@d.co", name="Ben").status_code == 302
    assert b"Planner" in client.get("/").data
    client.post("/logout", data={"_csrf": token(client, "/")})
    assert b"already exists" in register(client).data
    assert b"at least 8 characters" in register(client, email="e@f.co", password="short").data


def test_wrong_password_and_open_redirect(client):
    register(client)
    client.post("/logout", data={"_csrf": token(client, "/")})
    bad = client.post("/login", data={"_csrf": token(client), "email": "a@b.co", "password": "nope"})
    assert b"incorrect" in bad.data
    ok = client.post("/login?next=//evil.example", data={"_csrf": token(client), "email": "a@b.co", "password": "longenough1"})
    assert ok.status_code == 302 and ok.headers["Location"] == "/"


# ---------- demo mode ----------
@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setattr(data_source, "IS_DEMO", True)
    monkeypatch.setattr(data_source, "SKUS", [f"P{i:04d}" for i in range(1, 21)])


def test_demo_api_shapes(client, demo):
    register(client)
    d = client.get("/api/dashboard").get_json()
    assert d["kpis"]["analysed"] == d["kpis"]["sku_count"] == 20
    f = client.get("/api/forecast/P0001?horizon=7").get_json()
    n = len(f["dates"])
    assert len(f["forecast"]) == len(f["lower"]) == len(f["upper"]) == len(f["actual"]) == n
    assert all(lo <= fc <= hi for lo, fc, hi in zip(f["lower"], f["forecast"], f["upper"]))
    assert f["metrics"]["horizon_days"] == 7
    o = client.get("/api/optimizer/P0001").get_json()
    assert o["reorder_point"] >= o["safety_stock"] > 0 and o["projected"] is False
    assert client.get("/api/forecast/NOPE").get_json()["error"]


def test_param_validation_returns_clear_json_errors(client, demo):
    register(client)
    r = client.get("/api/optimizer/P0001?warehouse_capacity=abc")
    assert r.status_code == 400 and "Warehouse capacity" in r.get_json()["error"]
    r = client.get("/api/forecast/P0001?horizon=5")
    assert r.status_code == 400 and "7, 14 or 30" in r.get_json()["error"]
    r = client.get("/api/optimizer/P0001?lead_time_days=-3")
    assert r.status_code == 400
    assert client.get("/optimizer?budget=zzz").status_code == 200  # pages fall back to defaults


# ---------- real mode (with the team's schemas but fake model code) ----------
@pytest.fixture
def real(tmp_path, monkeypatch):
    pytest.importorskip("shared.schemas")
    import pandas as pd
    from shared.schemas import ForecastOutput, OptimizationConstraints, OptimizationOutput

    rows = []
    for sku_p in ("P0001", "P0002"):
        for d in pd.date_range("2024-01-01", periods=60):
            rows.append({"Date": d.date().isoformat(), "Store ID": "S001", "Product ID": sku_p,
                         "Units Sold": 100, "Inventory Level": 230})
    csv = tmp_path / "d.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    calls = {"forecast": 0, "optimize": 0}

    def load_csv(path):
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df["sku_id"] = df["Store ID"] + "::" + df["Product ID"]
        return df.sort_values(["sku_id", "Date"]).reset_index(drop=True)

    def forecast_demand(sku, h):
        calls["forecast"] += 1
        return SimpleNamespace(point_forecast=[100.0] * h, lower_bound=[80.0] * h,
                               upper_bound=[120.0] * h, model_confidence=0.8)

    def optimize_inventory(fc, c):
        calls["optimize"] += 1
        return OptimizationOutput(order_quantity=300, safety_stock=40, reorder_point=150,
                                  total_expected_cost=1234.5, solver_status="optimal")

    mods = {"forecast_demand": forecast_demand, "set_data_source": lambda df: None, "load_csv": load_csv,
            "optimize_inventory": optimize_inventory, "ForecastOutput": ForecastOutput,
            "OptimizationConstraints": OptimizationConstraints, "OptimizationOutput": OptimizationOutput}
    monkeypatch.setattr(real_backend, "_load_modules", lambda: mods)
    monkeypatch.setattr(real_backend, "CSV_PATH", str(csv))
    monkeypatch.setattr(real_backend, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(real_backend, "_df", None)
    monkeypatch.setattr(real_backend, "_cache_only", False)    # registered here so it is always restored after the test
    monkeypatch.setattr(data_source, "IS_DEMO", False)
    monkeypatch.setattr(data_source, "SKUS", real_backend.skus())
    return calls


def test_real_forecast_shape_and_caching(client, real):
    register(client)
    f = client.get("/api/forecast/S001::P0001?horizon=14").get_json()
    assert f["split_index"] == 45 and len(f["dates"]) == 45 + 14
    assert all(a is None for a in f["actual"][45:]) and all(v is None for v in f["forecast"][:45])
    assert f["metrics"]["confidence_pct"] == 80.0 and f["metrics"]["range_pct"] == 20.0
    assert f["metrics"]["certainty_pct"] == 80.0 and (f["metrics"]["lower_avg"], f["metrics"]["upper_avg"]) == (80, 120)
    # a second call and an optimizer call with different settings must not retrain the model
    client.get("/api/forecast/S001::P0001?horizon=14")
    client.get("/api/optimizer/S001::P0001?horizon=14&budget=9000")
    assert real["forecast"] == 1


def test_real_optimizer_uses_real_stock_and_reports_solver(client, real):
    register(client)
    o = client.get("/api/optimizer/S001::P0001").get_json()
    assert o["projected"] is True and o["stock_now"] == 230.0
    assert (o["order_qty"], o["reorder_point"], o["safety_stock"]) == (300, 150, 40)
    assert o["total_cost"] == 1234.5 and o["solver_status"] == "optimal"
    assert len(o["stock"]) == len(o["dates"]) == 14
    assert o["status"] == "Healthy"  # 230 is above 1.3 x reorder point (195)


def test_explanation_cached_only_on_success(client, real, monkeypatch):
    register(client)
    outcomes = iter([RuntimeError("no API key"), {"explanation": "Order 300 units.", "sources": ["doc1"]}])
    calls = []

    def generate_explanation(sku, fc, opt):
        calls.append(sku)
        out = next(outcomes)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setitem(sys.modules, "llm_orchestration", types.SimpleNamespace(generate_explanation=generate_explanation))
    first = client.get("/api/optimizer/S001::P0001").get_json()["explanation"]
    assert first["explanation"] is None and "no API key" in first["error"]
    second = client.get("/api/optimizer/S001::P0001").get_json()["explanation"]
    assert second["explanation"] == "Order 300 units." and second["sources"] == ["doc1"]
    client.get("/api/optimizer/S001::P0001")
    assert len(calls) == 2  # successful text is cached, so no third call


def test_real_portfolio_counts_only_default_settings(client, real):
    register(client)
    empty = client.get("/api/dashboard").get_json()
    assert empty["kpis"]["analysed"] == 0 and empty["kpis"]["sku_count"] == 2
    client.get("/api/optimizer/S001::P0001")
    client.get("/api/optimizer/S001::P0002")
    client.get("/api/optimizer/S001::P0002?budget=9999")  # custom settings are excluded
    d = client.get("/api/dashboard").get_json()
    assert d["kpis"]["analysed"] == 2 and d["kpis"]["total_cost"] == 2469
    assert d["kpis"]["avg_certainty_pct"] == 80.0 and sum(d["status_counts"].values()) == 2
    assert len(d["alerts"]) == 2 and d["alerts"][0]["days_of_cover"] <= d["alerts"][1]["days_of_cover"]
    assert {r["sku"] for r in d["top_orders"]} == {"S001::P0001", "S001::P0002"}


def test_policy_simulation():
    from web.services import policy
    r = policy.simulate(stock0=100, demand=[40] * 6, reorder_point=50, order_qty=200, lead_days=2)
    assert r["orders"] >= 1 and r["stock"][0] == 60.0 and 0 <= r["service_pct"] <= 100


def test_real_pages_render_with_settings_form(client, real):
    register(client)
    page = client.get("/optimizer?sku=S001::P0002&horizon=30&budget=7000").get_data(as_text=True)
    assert 'name="warehouse_capacity"' in page and 'value="7000"' in page and "S001::P0002" in page
    assert "Demo data" not in page
    assert client.get("/forecast?sku=S001::P0001").status_code == 200


# ---------- new pages ----------
def test_experiments_api_missing_then_present(client, app, tmp_path, demo):
    register(client)
    app.config["EXPERIMENTS_PATH"] = str(tmp_path / "none.json")
    r = client.get("/api/experiments")
    assert r.status_code == 404 and "scripts/run_experiments.py" in r.get_json()["error"]
    f = tmp_path / "e.json"
    f.write_text('{"datasets": {"x": {"name": "X"}}}')
    app.config["EXPERIMENTS_PATH"] = str(f)
    assert client.get("/api/experiments").get_json()["datasets"]["x"]["name"] == "X"
    assert client.get("/experiments").status_code == 200


def test_shipped_experiment_results_are_complete():
    import json, os
    path = "results/experiments.json"
    if not os.path.exists(path):
        pytest.skip("run scripts/run_experiments.py first")
    d = json.load(open(path))
    assert set(d["datasets"]) == {"real", "synthetic", "volatile"}
    for ds in d["datasets"].values():
        for sc in ("ample", "tight"):
            assert {"Simple rule", "CVXPY optimizer", "RL correction"} <= set(ds["policies"][sc])
        assert ds["split"]["test_days"] > 0 and ds["forecast_quality"][1]["coverage_pct"] is not None


def test_whatif_api_validation(client, demo):
    register(client)
    from scm import whatif
    sku = whatif.sku_list()[0]
    ok = client.get(f"/api/whatif/{sku}?demand_change=20&delay=2&capacity=")
    assert ok.status_code == 200 and "CVXPY optimizer" in ok.get_json()["policies"]
    assert client.get(f"/api/whatif/{sku}?delay=99").status_code == 400
    assert client.get(f"/api/whatif/{sku}?demand_change=abc").status_code == 400
    assert client.get("/api/whatif/NOPE").status_code == 404


def test_override_log(client, demo):
    register(client)
    sku = data_source.SKUS[0]
    def post(**kw):
        body = {"_csrf": token(client, "/optimizer"), "sku": sku, "recommended_qty": "300", "chosen_qty": "250",
                "reason": "Supplier promo", "next": "/optimizer?sku=" + sku}
        body.update(kw)
        return client.post("/overrides", data=body, follow_redirects=True)
    assert b"Saved to the change log" in post().data
    page = client.get("/optimizer?sku=" + sku).data
    assert b"Supplier promo" in page and b"Asha" in page
    assert b"Add a short reason" in post(reason="  ").data
    assert b"must be a number" in post(chosen_qty="abc").data or b"Enter the quantity" in post(chosen_qty="abc").data


def test_explanation_check_is_attached(client, real, monkeypatch):
    register(client)
    out = {"explanation": "Order 300 units; demand may reach 9999.", "sources": []}
    monkeypatch.setitem(sys.modules, "llm_orchestration", types.SimpleNamespace(generate_explanation=lambda *a: out))
    ex = client.get("/api/optimizer/S001::P0001").get_json()["explanation"]
    assert ex["check"]["ok"] is False and ex["check"]["unmatched"] == [9999.0]


# ---------- public view, rate limit, alerts, RL page, cache-only mode ----------
@pytest.fixture
def public_client(tmp_path, demo):
    from web import views
    views._hits.clear()
    app = create_app({"TESTING": True, "SECRET_KEY": "test", "DATABASE": str(tmp_path / "p.db"), "PUBLIC_VIEW": True})
    return app.test_client()


def test_public_view_open_pages_but_writes_need_login(public_client):
    for path in ("/", "/alerts", "/rl", "/experiments", "/forecast", "/optimizer", "/whatif", "/api/dashboard", "/api/alerts"):
        assert public_client.get(path).status_code in (200, 404), path   # 404 only for /api/experiments-style missing files
    assert public_client.get("/api/dashboard").status_code == 200
    assert b"Sign in" in public_client.get("/").data
    public_client.get("/login")
    with public_client.session_transaction() as s:
        t = s["_csrf"]
    r = public_client.post("/overrides", data={"_csrf": t, "sku": "P0001", "recommended_qty": "1", "chosen_qty": "2", "reason": "x"})
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_private_mode_still_requires_login(client, demo):
    assert client.get("/alerts").status_code == 302
    assert client.get("/api/alerts").status_code == 401


def test_whatif_rate_limit(public_client):
    from scm import whatif
    sku = whatif.sku_list()[0]
    codes = [public_client.get(f"/api/whatif/{sku}?delay=1").status_code for _ in range(22)]
    assert codes[:20] == [200] * 20 and 429 in codes[20:]


def test_alerts_sorted_and_have_actions(client, demo):
    register(client)
    rows = client.get("/api/alerts").get_json()
    assert len(rows) == 20 and rows == sorted(rows, key=lambda r: r["days_of_cover"])
    assert {"sku", "status", "order_qty", "certainty_pct"} <= set(rows[0])


def test_anonymous_never_triggers_the_llm(tmp_path, real, monkeypatch):
    called = []
    monkeypatch.setitem(sys.modules, "llm_orchestration",
                        types.SimpleNamespace(generate_explanation=lambda *a: called.append(1) or {"explanation": "x", "sources": []}))
    from web import views
    views._hits.clear()
    app = create_app({"TESTING": True, "SECRET_KEY": "t", "DATABASE": str(tmp_path / "a.db"), "PUBLIC_VIEW": True})
    c = app.test_client()
    ex = c.get("/api/optimizer/S001::P0001").get_json()["explanation"]
    assert ex is None and not called


def test_cache_only_mode_serves_cached_skus_without_models(client, real, monkeypatch):
    register(client)
    client.get("/api/optimizer/S001::P0001")                       # analyse one SKU with the (fake) models
    def no_models():
        raise ImportError("torch not installed")
    monkeypatch.setattr(real_backend, "_load_modules", no_models)
    monkeypatch.setattr(real_backend, "_df", None)
    assert real_backend.available() is True and real_backend.cache_only() is True
    monkeypatch.setattr(data_source, "SKUS", real_backend.skus())
    assert data_source.SKUS == ["S001::P0001"]
    ok = client.get("/api/optimizer/S001::P0001").get_json()
    assert ok["order_qty"] == 300 and ok["stock_now"] == 230.0
    assert client.get("/api/forecast/S001::P0001").status_code == 200
    missing = client.get("/api/forecast/S001::P0002")
    assert missing.status_code == 404                                # not in SKUS, so a plain 404
    monkeypatch.setattr(data_source, "SKUS", ["S001::P0001", "S001::P0002"])
    r = client.get("/api/forecast/S001::P0002")
    assert r.status_code == 404 and "not been analysed" in r.get_json()["error"]


def test_admin_email_setting_controls_who_is_admin(tmp_path, demo):
    app = create_app({"TESTING": True, "SECRET_KEY": "t", "DATABASE": str(tmp_path / "x.db"), "ADMIN_EMAIL": "boss@x.co"})
    c = app.test_client()
    register(c, email="first@x.co")                       # first user, but NOT the configured admin
    assert b"Planner" in c.get("/").data
    c.post("/logout", data={"_csrf": token(c, "/")})
    register(c, email="boss@x.co", name="Boss")
    assert b"Admin" in c.get("/").data


# ---------- PDF reports ----------
def _pdf_text(data):
    pypdf = pytest.importorskip("pypdf")
    from io import BytesIO
    return " ".join(page.extract_text() for page in pypdf.PdfReader(BytesIO(data)).pages)


def test_portfolio_pdf(client, demo):
    pytest.importorskip("reportlab")
    assert client.get("/report.pdf").status_code == 302          # private mode: login first
    register(client)
    r = client.get("/report.pdf")
    assert r.status_code == 200 and r.mimetype == "application/pdf" and r.data.startswith(b"%PDF")
    assert "attachment" in r.headers["Content-Disposition"] and ".pdf" in r.headers["Content-Disposition"]
    text = _pdf_text(r.data)
    assert "Inventory report" in text and "DEMO DATA" in text and "What needs action" in text


def test_sku_pdf_has_bounds_and_validates_input(client, demo):
    pytest.importorskip("reportlab")
    register(client)
    r = client.get("/report/P0003.pdf?horizon=7")
    assert r.status_code == 200 and r.data.startswith(b"%PDF")
    text = _pdf_text(r.data)
    assert "Reorder plan: P0003" in text and "Lower bound" in text and "Upper bound" in text and "Order quantity" in text
    assert client.get("/report/NOPE.pdf").status_code == 404
    assert client.get("/report/P0003.pdf?horizon=5").status_code == 400


def test_real_sku_pdf_includes_cached_explanation_but_anonymous_cannot_trigger_llm(client, real, monkeypatch, tmp_path):
    pytest.importorskip("reportlab")
    register(client)
    calls = []
    monkeypatch.setitem(sys.modules, "llm_orchestration", types.SimpleNamespace(
        generate_explanation=lambda *a: calls.append(1) or {"explanation": "Order 300 units today.", "sources": []}))
    client.get("/api/optimizer/S001::P0001")                       # signed-in view generates and caches the text
    text = _pdf_text(client.get("/report/S001::P0001.pdf").data)
    assert "Why this plan" in text and "Order 300 units today." in text and "DEMO DATA" not in text
    from web import views
    views._hits.clear()
    anon = create_app({"TESTING": True, "SECRET_KEY": "t", "DATABASE": str(tmp_path / "r.db"), "PUBLIC_VIEW": True}).test_client()
    before = len(calls)
    assert anon.get("/report/S001::P0001.pdf").status_code == 200
    assert len(calls) == before                                    # reading a report never calls the LLM


def test_rate_limits_are_separate_per_feature(public_client):
    pytest.importorskip("reportlab")
    from scm import whatif
    sku = whatif.sku_list()[0]
    for _ in range(20):
        assert public_client.get(f"/api/whatif/{sku}?delay=1").status_code == 200
    assert public_client.get(f"/api/whatif/{sku}?delay=1").status_code == 429
    assert public_client.get("/report.pdf").status_code == 200          # reports have their own allowance
