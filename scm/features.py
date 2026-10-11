"""Features known at decision time t (information up to day t-1, plus promotions planned in advance).

The target is the average daily demand over the protection interval t..t+H-1, with H = lead time + 1,
divided by each SKU's training-set mean so one pooled model serves every SKU.
"""
import numpy as np
import pandas as pd

MIN_HISTORY = 28


def _shift(a, k):
    out = np.full_like(a, np.nan)
    out[k:] = a[:-k]
    return out


def _rolling_prev_mean(a, w):
    """Mean of a[t-w:t] (excludes day t)."""
    cs = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
    out = np.full_like(a, np.nan)
    T = a.shape[0]
    out[w:] = (cs[w:T] - cs[0:T - w]) / w
    return out


def _forward_mean(a, H):
    """Mean of a[t:t+H], truncated at the end of the series."""
    T = a.shape[0]
    cs = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
    hi = np.minimum(np.arange(T) + H, T)
    return (cs[hi] - cs[np.arange(T)]) / (hi - np.arange(T))[:, None]


def build(panel, H, train_end):
    """Return dict with X (T,N,F), y (T,N) normalized target, scale (N,), and the feature names."""
    d = panel.demand
    scale = d[:train_end].mean(axis=0)
    dn = d / scale
    T, N = d.shape
    dates = pd.DatetimeIndex(panel.dates)
    dow = np.eye(7)[dates.dayofweek.to_numpy()]                  # (T,7)
    doy = dates.dayofyear.to_numpy()
    cols = [
        ("lag1", _shift(dn, 1)), ("lag7", _shift(dn, 7)), ("lag14", _shift(dn, 14)),
        ("roll7", _rolling_prev_mean(dn, 7)), ("roll28", _rolling_prev_mean(dn, 28)),
        ("promo_plan", _forward_mean(panel.promo, H)), ("discount_plan", _forward_mean(panel.discount, H) / 20.0),
    ]
    feats = [c for _, c in cols]
    names = [n for n, _ in cols]
    for k in range(7):
        feats.append(np.repeat(dow[:, [k]], N, axis=1))
        names.append(f"dow{k}")
    feats.append(np.repeat(np.sin(2 * np.pi * doy / 365)[:, None], N, axis=1)); names.append("sin_year")
    feats.append(np.repeat(np.cos(2 * np.pi * doy / 365)[:, None], N, axis=1)); names.append("cos_year")
    X = np.stack(feats, axis=2)                                    # (T,N,F)

    y = np.full((T, N), np.nan)
    for t in range(T - H + 1):
        y[t] = dn[t:t + H].mean(axis=0)
    return {"X": X, "y": y, "scale": scale, "names": names, "H": H}


def rows(feat, lo, hi, need_target=True):
    """Pooled (rows, F) design matrix and target for decision days t in [lo, hi)."""
    T = feat["X"].shape[0]
    ts = np.arange(max(lo, MIN_HISTORY), hi)
    if need_target:
        ts = ts[ts <= T - feat["H"]]
    X = feat["X"][ts].reshape(-1, feat["X"].shape[2])
    y = feat["y"][ts].reshape(-1) if need_target else None
    return X, y, ts
