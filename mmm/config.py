"""Central configuration. Everything a business user may want to change lives here."""
import os
from typing import List, Optional
from pydantic import BaseModel


class MMMConfig(BaseModel):
    # ---- data -------------------------------------------------------------------------------
    date_column: str = "DATE"
    target_column: str = "revenue"
    spend_columns: List[str] = ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"]
    # Continuous exogenous controls (z-scored internally). Customer counts are NOT controls:
    # they are outcomes of marketing, so conditioning on them would bias channel effects.
    control_columns: List[str] = ["competitor_sales_B", "newsletter"]
    event_column: Optional[str] = "events"        # categorical -> one 0/1 dummy per event
    event_none_label: str = "na"
    add_trend: bool = True                        # linear trend control (z-scored week index)
    data_path: str = os.environ.get("MMM_DATA_PATH", "data/dt_simulated_weekly.csv")
    artifacts_dir: str = os.environ.get("MMM_ARTIFACTS_DIR", "artifacts")
    currency: str = "$"

    # ---- model ------------------------------------------------------------------------------
    adstock_l_max: int = 8                        # weeks of carry-over considered
    yearly_seasonality: int = 2                   # Fourier order
    # Prior on channel effect size: HalfNormal(sigma = scale * n_channels * spend_share).
    # Bigger spenders are allowed bigger effects a priori, without forcing it.
    beta_prior_scale: float = 0.5
    control_prior_sigma: float = 0.5

    # ---- sampling ---------------------------------------------------------------------------
    draws: int = 2000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.95
    seed: int = 42

    # ---- validation -------------------------------------------------------------------------
    cv_folds: int = 3
    cv_test_size: int = 13                        # weeks per hold-out window
    rhat_max: float = 1.01
    ess_min: float = 400.0
    divergence_max_frac: float = 0.01
    max_mape: float = 0.15                        # acceptance threshold for out-of-sample error
    breakeven_roas: float = 1.0                   # revenue ROAS needed to cover cost (1 / gross margin)

    # ---- optimizer --------------------------------------------------------------------------
    min_spend_mult: float = 0.5                   # per-channel floor vs current average weekly spend
    max_spend_mult: float = 2.0                   # per-channel cap  vs current average weekly spend
    risk_aversion: float = 0.0                    # 0 = maximise expected revenue; >0 penalises uncertainty
    interval: float = 0.90                        # credible interval width used everywhere


CFG = MMMConfig()
