"""Synthetic MMM data with KNOWN ground truth, used to prove the pipeline recovers true ROAS."""
from typing import Tuple
import numpy as np
import pandas as pd

from .analysis import geometric_adstock, logistic_saturation
from .config import MMMConfig


def simulate(n_weeks: int = 156, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, MMMConfig]:
    rng = np.random.default_rng(seed)
    channels = ["tv", "search", "social"]
    alpha = np.array([0.6, 0.2, 0.4])
    lam = np.array([2.0, 4.0, 3.0])
    amp = np.array([150_000.0, 80_000.0, 60_000.0])     # max weekly revenue each channel can add
    l_max = 8

    dates = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    spend = np.zeros((n_weeks, 3))
    for c, base in enumerate([40_000, 15_000, 10_000]):
        on = rng.random(n_weeks) < 0.7                       # flighting: some weeks off
        spend[:, c] = on * base * rng.lognormal(0, 0.45, n_weeks)

    season = 40_000 * np.sin(2 * np.pi * np.arange(n_weeks) / 52.18)
    contrib = np.zeros((n_weeks, 3))
    for c in range(3):
        ad = geometric_adstock(spend[:, c] / spend[:, c].max(), np.array([alpha[c]]), l_max)[0]
        contrib[:, c] = amp[c] * logistic_saturation(ad, lam[c])
    revenue = 600_000 + season + contrib.sum(axis=1) + rng.normal(0, 15_000, n_weeks)

    df = pd.DataFrame({"DATE": dates, "revenue": revenue})
    for c, ch in enumerate(channels):
        df[f"{ch}_S"] = spend[:, c]
    truth = pd.DataFrame({"channel": [f"{c}_S" for c in channels],
                          "true_roas": contrib.sum(axis=0) / spend.sum(axis=0),
                          "true_alpha": alpha, "true_lam": lam})
    cfg = MMMConfig(spend_columns=[f"{c}_S" for c in channels], control_columns=[], event_column=None,
                    chains=4, draws=1000, tune=1000, cv_folds=2, cv_test_size=13)
    return df, truth, cfg
