"""Bayesian MMM construction and fitting (pymc-marketing >= 1.3)."""
import os
from typing import Optional
import numpy as np
import pandas as pd
from pymc_extras.prior import Prior
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from .config import CFG, MMMConfig

PARAM_VARS = ["adstock_alpha", "saturation_lam", "saturation_beta", "gamma_control",
              "gamma_fourier", "intercept_contribution", "y_sigma"]


def build_mmm(X: pd.DataFrame, cfg: MMMConfig = CFG) -> MMM:
    """Geometric adstock -> logistic saturation, Fourier seasonality, standardised controls.

    Priors are weakly informative and spend-aware: the sigma of each channel's effect size is
    proportional to its share of spend (the standard 'spend-share prior' used in industry MMMs)."""
    spend = X[cfg.spend_columns].sum()
    share = (spend / spend.sum()).values
    beta_sigma = np.clip(cfg.beta_prior_scale * len(share) * share, 0.05, None)
    controls = [c for c in X.columns if c not in [cfg.date_column, cfg.target_column] + cfg.spend_columns]
    model_config = {
        "saturation_beta": Prior("HalfNormal", sigma=beta_sigma, dims="channel"),
        "gamma_control": Prior("Normal", mu=0, sigma=cfg.control_prior_sigma, dims="control"),
    }
    return MMM(
        date_column=cfg.date_column,
        channel_columns=cfg.spend_columns,
        target_column=cfg.target_column,
        control_columns=controls or None,
        adstock=GeometricAdstock(l_max=cfg.adstock_l_max),
        saturation=LogisticSaturation(),
        yearly_seasonality=cfg.yearly_seasonality,
        model_config=model_config,
    )


def fit_mmm(mmm: MMM, X: pd.DataFrame, y: pd.Series, cfg: MMMConfig = CFG,
            draws: Optional[int] = None, tune: Optional[int] = None, chains: Optional[int] = None) -> MMM:
    chains = chains or cfg.chains
    mmm.fit(
        X, y,
        nuts_sampler="nutpie",
        draws=draws or cfg.draws,
        tune=tune or cfg.tune,
        chains=chains,
        cores=max(1, min(chains, os.cpu_count() or 1)),
        target_accept=cfg.target_accept,
        random_seed=cfg.seed,
        progressbar=False,
    )
    return mmm


def add_lift_tests(mmm: MMM, lift_df: pd.DataFrame) -> MMM:
    """Calibrate the saturation curves with experiment results (geo-lift / holdout tests).

    lift_df columns: channel, x (baseline spend), delta_x (spend change), delta_y (observed revenue
    change), sigma (standard error of delta_y). Must be called after build and BEFORE fit."""
    mmm.add_lift_test_measurements(lift_df)
    return mmm
