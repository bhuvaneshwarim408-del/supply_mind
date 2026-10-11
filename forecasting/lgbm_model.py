"""
forecasting/lgbm_model.py

LGBMRegressor with quantile objective (objective='quantile'), trained
three times at alpha = 0.1, 0.5, 0.9 to yield point + interval forecast
without a separate uncertainty model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import optuna
from lightgbm import LGBMRegressor

QUANTILES = {"lower": 0.1, "median": 0.5, "upper": 0.9}

DEFAULT_PARAMS = dict(
    objective="quantile",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=0.0,
    verbosity=-1,
    random_state=42,
)


@dataclass
class LGBMQuantileForecaster:
    """Wraps three LGBMRegressor models, one per quantile."""

    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    models: dict = field(default_factory=dict)

    def fit(self, X, y, eval_set=None, early_stopping_rounds: int = 50):
        callbacks = None
        if eval_set is not None:
            import lightgbm as lgb
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

        for name, alpha in QUANTILES.items():
            p = dict(self.params)
            p["alpha"] = alpha
            model = LGBMRegressor(**p)
            if eval_set is not None:
                model.fit(X, y, eval_set=[eval_set], callbacks=callbacks)
            else:
                model.fit(X, y)
            self.models[name] = model
        return self

    def predict(self, X) -> dict[str, np.ndarray]:
        if not self.models:
            raise RuntimeError("Call .fit() before .predict().")
        preds = {name: model.predict(X) for name, model in self.models.items()}
        # enforce monotonicity: lower <= median <= upper
        lower = np.minimum(preds["lower"], preds["median"])
        upper = np.maximum(preds["upper"], preds["median"])
        return {"lower": lower, "median": preds["median"], "upper": upper}

    def tune(
        self,
        X_train, y_train, X_val, y_val,
        n_trials: int = 30,
        alpha: float = 0.5,
        timeout: int | None = None,
    ) -> dict:
        """
        Optuna hyperparameter search on the median (alpha=0.5) model;
        the tuned params are then reused for the 0.1 / 0.9 quantile
        models (only `alpha` differs between them).
        """

        def objective(trial: optuna.Trial) -> float:
            p = dict(
                objective="quantile",
                alpha=alpha,
                n_estimators=trial.suggest_int("n_estimators", 100, 800),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                verbosity=-1,
                random_state=42,
            )
            model = LGBMRegressor(**p)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            # pinball / quantile loss
            diff = y_val - pred
            loss = np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))
            return loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best = dict(DEFAULT_PARAMS)
        best.update(study.best_params)
        self.params = best
        return study.best_params
