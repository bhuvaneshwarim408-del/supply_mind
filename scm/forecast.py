"""Forecasters for the protection-interval demand.

MSEForecaster       ridge regression trained on squared error, with split-conformal quantiles for safety stock.
DecisionFocused     the same linear model, trained directly on the newsvendor cost (holding vs stockout).
For a single period, minimizing this cost is the same as quantile regression at p/(p+h); the experiments
test whether that helps once the policy is run in the full simulator.
"""
import numpy as np


def critical_ratio(holding, stockout):
    return stockout / (stockout + holding)


def conformal_quantile(residuals, tau):
    """Split-conformal quantile: finite-sample corrected so coverage is at least tau on exchangeable data."""
    r = np.sort(np.asarray(residuals).ravel())
    n = len(r)
    k = int(np.ceil((n + 1) * tau)) - 1
    return float(r[min(max(k, 0), n - 1)])


class Standardizer:
    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0
        keep_const = self.sd == 1.0
        self.mu[keep_const & (X.std(axis=0) < 1e-9)] = 0.0
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


def _clean(X, y=None):
    X = np.nan_to_num(X, nan=1.0)  # missing history early on: neutral value in normalized units
    return X if y is None else (X, y)


class MSEForecaster:
    def __init__(self, lam=10.0):
        self.lam = lam

    def fit(self, X, y):
        X = _clean(X)
        self.std = Standardizer().fit(X)
        Z = np.hstack([self.std(X), np.ones((len(X), 1))])
        reg = self.lam * np.eye(Z.shape[1])
        reg[-1, -1] = 0.0
        self.w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
        return self

    def predict(self, X):
        X = _clean(X)
        return np.hstack([self.std(X), np.ones((len(X), 1))]) @ self.w

    def calibrate(self, X_val, y_val):
        """Signed residuals on held-out validation rows (the conformal calibration set)."""
        self.resid = y_val - self.predict(X_val)
        return self

    def quantile(self, X, tau):
        return self.predict(X) + conformal_quantile(self.resid, tau)

    def interval(self, X, coverage=0.8):
        a = (1 - coverage) / 2
        p = self.predict(X)
        return p + conformal_quantile(self.resid, a), p + conformal_quantile(self.resid, 1 - a)


class DecisionFocused:
    """Linear order-up-to model trained directly on the newsvendor cost  h*(S-y)+ + p*(y-S)+  (normalized units).

    The training problem is convex, so it is solved exactly with CVXPY (no learning rate, no early stopping).
    For one period this equals quantile regression at p/(p+h), which is why it is a fair 'decision-focused' baseline.
    """

    def __init__(self, holding, stockout, l2=1e-3):
        self.h, self.p, self.l2 = holding, stockout, l2

    def fit(self, X, y, X_val=None, y_val=None, init: MSEForecaster = None):
        import cvxpy as cp
        X = _clean(X)
        self.std = init.std if init is not None else Standardizer().fit(X)
        Z = np.hstack([self.std(X), np.ones((len(X), 1))])
        w = cp.Variable(Z.shape[1])
        S = Z @ w
        loss = cp.sum(self.h * cp.pos(S - y) + self.p * cp.pos(y - S)) / len(Z) + self.l2 * cp.sum_squares(w[:-1])
        cp.Problem(cp.Minimize(loss)).solve(solver=cp.CLARABEL)
        self.w = np.asarray(w.value)
        return self

    def predict(self, X):
        X = _clean(X)
        return np.hstack([self.std(X), np.ones((len(X), 1))]) @ self.w


def newsvendor_cost(S, y, holding, stockout):
    return float(np.mean(holding * np.maximum(S - y, 0) + stockout * np.maximum(y - S, 0)))


def wape(pred, y):
    return float(np.abs(pred - y).sum() / np.abs(y).sum())
