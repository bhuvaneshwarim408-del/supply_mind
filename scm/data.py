"""Demand panels: the retail CSV, and a structured synthetic generator for controlled experiments."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Panel:
    name: str
    dates: np.ndarray       # (T,) datetime64
    skus: list              # N ids
    demand: np.ndarray      # (T, N) units sold
    promo: np.ndarray       # (T, N) 0/1, planned in advance
    discount: np.ndarray    # (T, N) percent, planned in advance
    unit_value: np.ndarray = None   # (N,) relative price of each SKU (mean 1); scales its holding and stockout cost

    @property
    def T(self):
        return self.demand.shape[0]

    @property
    def N(self):
        return self.demand.shape[1]

    def split(self, train=0.70, val=0.15):
        """Time-ordered split points shared by every SKU: train < val < test. No shuffling."""
        a = int(self.T * train)
        b = int(self.T * (train + val))
        return a, b


def load_panel(csv_path, n_skus=None, name="retail csv"):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["sku_id"] = df["Store ID"] + "::" + df["Product ID"]

    def piv(col):
        return df.pivot(index="Date", columns="sku_id", values=col).sort_index()

    d, pr, dc, price = piv("Units Sold"), piv("Holiday/Promotion"), piv("Discount"), piv("Price")
    skus = list(d.columns)
    if n_skus and n_skus < len(skus):
        skus = [skus[i] for i in np.linspace(0, len(skus) - 1, n_skus).astype(int)]
    uv = price[skus].mean().to_numpy(float)
    return Panel(name, d.index.values, skus, d[skus].to_numpy(float), pr[skus].to_numpy(float), dc[skus].to_numpy(float), uv / uv.mean())


def synthetic_panel(n_skus=40, days=731, seed=0, name="structured synthetic", volatile_promo=False):
    """Demand WITH signal: weekly pattern, yearly season, promotions and noise.

    Used to test whether forecasting and decision-focused training help when there is something to learn.
    volatile_promo=True makes promotion days much less predictable (more noise), a setting where a fixed
    safety margin is wrong and a cost-trained model can adapt.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=days).values
    dow = pd.DatetimeIndex(dates).dayofweek.to_numpy()
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    weekly = np.array([0.75, 0.8, 0.9, 1.0, 1.25, 1.6, 1.3])
    base = rng.lognormal(np.log(110), 0.35, n_skus)
    amp = rng.uniform(0.05, 0.3, n_skus)
    phase = rng.uniform(0, 2 * np.pi, n_skus)
    promo = np.zeros((days, n_skus))
    for i in range(n_skus):
        for s in rng.choice(days - 4, size=days // 14, replace=False):
            promo[s:s + rng.integers(2, 5), i] = 1.0
    discount = promo * rng.choice([10, 15, 20], size=(days, n_skus))
    mean = (base[None, :] * weekly[dow][:, None]
            * (1 + amp[None, :] * np.sin(2 * np.pi * doy[:, None] / 365 + phase[None, :]))
            * (1 + 0.6 * promo) * (1 + 0.0004 * np.arange(days))[:, None])
    shape = np.where(promo > 0, 2.5, 12.0) if volatile_promo else np.full(mean.shape, 12.0)
    gamma_noise = rng.gamma(shape=shape, scale=1 / shape)  # about 29% variability (63% on volatile promo days)
    demand = rng.poisson(mean * gamma_noise).astype(float)
    value = rng.lognormal(0.0, 0.6, n_skus)   # some SKUs are much more valuable per unit than others
    return Panel(name, dates, [f"SYN::P{i:04d}" for i in range(n_skus)], demand, promo, discount, value / value.mean())
