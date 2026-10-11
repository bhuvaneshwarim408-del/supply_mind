import numpy as np
import pytest

from scm import data, faithfulness, features, forecast
from scm.policies import BaseStockPolicy, CvxpyPolicy
from scm.simulator import SimParams, apply_capacity, simulate


@pytest.fixture(scope="module")
def synth():
    return data.synthetic_panel(n_skus=8, days=400, seed=3)


def test_split_is_ordered_and_disjoint(synth):
    a, b = synth.split()
    assert 0 < a < b < synth.T


def test_features_use_only_the_past(synth):
    """Changing demand on day t or later must not change the features for day t."""
    a, _ = synth.split()
    H = 4
    F1 = features.build(synth, H, a)
    shifted = data.synthetic_panel(n_skus=8, days=400, seed=3)
    t = a + 20                 # after the training period, so the per-SKU scale (a training statistic) is unchanged
    shifted.demand[t:] *= 3.0
    F2 = features.build(shifted, H, a)
    assert np.allclose(F1["X"][t], F2["X"][t], equal_nan=True)


def test_conformal_coverage_close_to_nominal():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(size=5000), rng.normal(size=20000)
    q = forecast.conformal_quantile(cal, 0.8)
    assert abs(np.mean(test <= q) - 0.8) < 0.02


def test_decision_focused_fit_is_a_quantile_regression(synth):
    a, b = synth.split()
    F = features.build(synth, 4, a)
    X, y, _ = features.rows(F, 0, a)
    mse = forecast.MSEForecaster().fit(X, y)
    tau = 0.8
    dfl = forecast.DecisionFocused(holding=1.0, stockout=4.0).fit(X, y, init=mse)   # ratio 4/(4+1) = 0.8
    assert abs(np.mean(y <= dfl.predict(X)) - tau) < 0.03


def test_simulator_accounting_by_hand():
    demand = np.array([[5.0], [5.0], [5.0]])
    class Fixed:  # order 10 on day 0 only
        def order(self, t, on_hand, pos):
            return np.array([10.0]) if t == 0 else np.array([0.0])
    r = simulate(demand, 0, 3, Fixed(), SimParams(holding=1.0, stockout=10.0, lead=1), np.array([5.0]))
    # day0: stock 5, sell 5 -> 0 left. day1: +10 arrives, sell 5 -> 5 left (hold 5). day2: sell 5 -> 0.
    assert r.holding_cost == 5.0 and r.stockout_cost == 0.0 and r.fill_rate == 1.0


def test_capacity_scaling_respects_limit():
    q = apply_capacity(np.array([60.0, 40.0]), pos=np.array([20.0, 10.0]), capacity=100.0)
    assert q.sum() == pytest.approx(70.0) and q[0] / q[1] == pytest.approx(1.5)
    assert np.allclose(apply_capacity(np.array([5.0, 5.0]), np.array([0.0, 0.0]), None), [5.0, 5.0])


def _cv(n=1, capacity=None, n_scen=200):
    scale = np.full(n, 100.0)
    resid = np.random.default_rng(1).normal(0, 0.3, 4000)
    return CvxpyPolicy(np.ones((1, n)), scale, 4, resid, holding=1.0, stockout=4.0, capacity=capacity, n_scen=n_scen), resid, scale


def test_newsvendor_check_lp_matches_the_quantile():
    """One SKU, no limits: the LP must order up to the p/(p+h) quantile of demand (the textbook answer)."""
    cv, resid, scale = _cv()
    q = cv.order(0, np.zeros(1), np.zeros(1))
    expected = (1.0 + np.quantile(resid, 0.8)) * 4 * 100.0
    assert q[0] == pytest.approx(expected, rel=0.02)


def test_shadow_price_positive_only_when_capacity_binds():
    tight, _, _ = _cv(n=2, capacity=500.0)
    tight.order(0, np.zeros(2), np.zeros(2))
    assert tight.shadow and tight.shadow[-1] > 0
    slack, _, _ = _cv(n=2, capacity=1e6)
    slack.order(0, np.zeros(2), np.zeros(2))
    assert slack.shadow[-1] == pytest.approx(0.0, abs=1e-6)


def test_lp_never_exceeds_capacity():
    cv, _, _ = _cv(n=3, capacity=700.0)
    pos = np.array([100.0, 50.0, 0.0])
    q = cv.order(0, pos, pos)
    assert (pos + q).sum() <= 700.0 + 1e-3


def test_faithfulness_check():
    allowed = [300, 150, 40, 80.0, 1234.5]
    ok = faithfulness.check("Order 300 units. Reorder at 150 with 40 safety stock; confidence is 80%.", allowed)
    assert ok["ok"] and ok["checked"] == 4
    bad = faithfulness.check("Order 300 units because sales will reach 725 next week.", allowed)
    assert not bad["ok"] and bad["unmatched"] == [725.0]
    assert faithfulness.check("Step 1 and 2: order 3 days ahead.", allowed)["ok"]   # small wording numbers are skipped


def test_whatif_runs_and_delay_hurts():
    from scm import whatif
    sku = whatif.sku_list()[0]
    calm = whatif.run(sku, 0, 0, None)
    shock = whatif.run(sku, 30, 3, None)
    assert set(calm["policies"]) == {"Simple rule", "Forecast base-stock", "CVXPY optimizer"}
    assert len(calm["dates"]) == len(calm["policies"]["Simple rule"]["stock"]) == 60
    assert shock["policies"]["Forecast base-stock"]["total_cost"] > calm["policies"]["Forecast base-stock"]["total_cost"]
