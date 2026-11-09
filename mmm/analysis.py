"""Posterior analytics: decomposition, ROAS / marginal ROAS, response curves, adstock half-life.

All business numbers are derived from posterior DRAWS, so every figure carries an uncertainty interval.
Response curves use closed-form transformations that are verified against the fitted model's own
`channel_contribution` in the test-suite (see tests/test_recovery.py)"""
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .config import CFG, MMMConfig


# ---------------------------------------------------------------------------------------------
# Transformations (numpy mirrors of pymc-marketing's GeometricAdstock + LogisticSaturation)
# ---------------------------------------------------------------------------------------------
def geometric_adstock(x: np.ndarray, alpha: np.ndarray, l_max: int, normalize: bool = True) -> np.ndarray:
    """x: [T] scaled spend, alpha: [S] decay draws -> adstocked series [S, T] (causal, zero-padded)."""
    x = np.asarray(x, float)
    alpha = np.atleast_1d(np.asarray(alpha, float))
    k = np.arange(l_max)
    w = alpha[:, None] ** k[None, :]
    if normalize:
        w = w / w.sum(axis=1, keepdims=True)
    T = len(x)
    out = np.zeros((len(alpha), T))
    for lag in range(l_max):
        out[:, lag:] += w[:, [lag]] * x[None, : T - lag]
    return out


def logistic_saturation(u: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """(1 - exp(-lam u)) / (1 + exp(-lam u)) == tanh(lam u / 2). Bounded in [0, 1)."""
    return np.tanh(np.asarray(lam) * np.asarray(u) / 2.0)


def half_life(alpha: np.ndarray) -> np.ndarray:
    """Weeks until a spend impulse has lost half of its (geometric) carry-over weight."""
    alpha = np.clip(np.asarray(alpha, float), 1e-9, 1 - 1e-9)
    return np.log(0.5) / np.log(alpha)


# ---------------------------------------------------------------------------------------------
# Parameter container (decoupled from PyMC so the dashboard / API stay light-weight)
# ---------------------------------------------------------------------------------------------
@dataclass
class Params:
    channels: List[str]
    alpha: np.ndarray            # [S, C]
    lam: np.ndarray              # [S, C]
    beta: np.ndarray             # [S, C]   (in scaled-target units)
    channel_scale: np.ndarray    # [C]
    target_scale: float
    l_max: int
    spend_mean: np.ndarray       # [C] historical average weekly spend
    spend_max: np.ndarray        # [C] historical max weekly spend
    n_weeks: int

    @property
    def n_draws(self) -> int:
        return self.alpha.shape[0]

    def save(self, path: str) -> None:
        np.savez_compressed(path, channels=np.array(self.channels), alpha=self.alpha, lam=self.lam,
                            beta=self.beta, channel_scale=self.channel_scale,
                            target_scale=self.target_scale, l_max=self.l_max,
                            spend_mean=self.spend_mean, spend_max=self.spend_max, n_weeks=self.n_weeks)

    @staticmethod
    def load(path: str) -> "Params":
        z = np.load(path, allow_pickle=False)
        return Params(list(z["channels"]), z["alpha"], z["lam"], z["beta"], z["channel_scale"],
                      float(z["target_scale"]), int(z["l_max"]), z["spend_mean"], z["spend_max"],
                      int(z["n_weeks"]))


def _node(idata, group: str):
    return idata[group]


def _flat(da, extra_dim: str = None) -> np.ndarray:
    """(chain, draw, [dim]) -> (S, [dim]) numpy."""
    s = da.stack(sample=("chain", "draw"))
    if extra_dim:
        s = s.transpose("sample", extra_dim)
    return np.asarray(s.values)


def extract_params(mmm, X: pd.DataFrame, cfg: MMMConfig = CFG) -> Params:
    post = _node(mmm.idata, "posterior")
    const = _node(mmm.idata, "constant_data")
    channels = list(cfg.spend_columns)
    spend = X[channels]

    # Handle target_scale and channel_scale robustly if they are already scalars or DataArrays
    t_scale_val = const["target_scale"]
    if hasattr(t_scale_val, "values"):
        t_scale_val = t_scale_val.values
    target_scale = float(np.atleast_1d(t_scale_val)[0])

    c_scale_val = const["channel_scale"]
    if hasattr(c_scale_val, "values"):
        c_scale_val = c_scale_val.values
    channel_scale = np.asarray(c_scale_val, float)

    return Params(
        channels=channels,
        alpha=_flat(post["adstock_alpha"], "channel"),
        lam=_flat(post["saturation_lam"], "channel"),
        beta=_flat(post["saturation_beta"], "channel"),
        channel_scale=channel_scale,
        target_scale=target_scale,
        l_max=cfg.adstock_l_max,
        spend_mean=spend.mean().values.astype(float),
        spend_max=spend.max().values.astype(float),
        n_weeks=len(X),
    )


# ---------------------------------------------------------------------------------------------
# Contributions (actual weekly history) and efficiency metrics
# ---------------------------------------------------------------------------------------------
def channel_contributions(P: Params, spend: np.ndarray) -> np.ndarray:
    """Weekly incremental revenue per channel in ORIGINAL units. spend: [T, C] -> [S, T, C]."""
    T, C = spend.shape
    out = np.zeros((P.n_draws, T, C))
    for c in range(C):
        ad = geometric_adstock(spend[:, c] / P.channel_scale[c], P.alpha[:, c], P.l_max)   # [S, T]
        out[:, :, c] = P.beta[:, [c]] * logistic_saturation(ad, P.lam[:, [c]]) * P.target_scale
    return out


def _ci(a: np.ndarray, axis=0, level: float = 0.90) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return np.quantile(a, lo, axis=axis), np.quantile(a, hi, axis=axis)


def summarise(a: np.ndarray, level: float = 0.90) -> Dict[str, float]:
    lo, hi = _ci(a, 0, level)
    return {"mean": float(np.mean(a)), "lo": float(lo), "hi": float(hi)}


def roas_table(P: Params, spend: np.ndarray, level: float = 0.90, cfg: MMMConfig = CFG) -> pd.DataFrame:
    """ROAS = incremental revenue / spend over the sample; mROAS = revenue from the NEXT dollar
    at the current average weekly spend; plus saturation and carry-over diagnostics."""
    contrib = channel_contributions(P, spend)                                   # [S, T, C]
    total_spend = spend.sum(axis=0)                                             # [C]
    roas = contrib.sum(axis=1) / np.maximum(total_spend, 1e-9)                  # [S, C]
    u = P.spend_mean / P.channel_scale                                          # [C]
    sech2 = 1.0 / np.cosh(P.lam * u / 2.0) ** 2
    mroas = P.beta * P.target_scale * (P.lam / 2.0) * sech2 / P.channel_scale   # [S, C]
    sat_level = logistic_saturation(u, P.lam)                                   # [S, C]
    hl = half_life(P.alpha)
    rows = []
    for c, ch in enumerate(P.channels):
        r, m, s, h = (summarise(roas[:, c], level), summarise(mroas[:, c], level),
                      summarise(sat_level[:, c], level), summarise(hl[:, c], level))
        rows.append(dict(
            channel=ch, spend=float(total_spend[c]),
            incremental_revenue=float(contrib[:, :, c].sum(axis=1).mean()),
            roas=r["mean"], roas_lo=r["lo"], roas_hi=r["hi"],
            mroas=m["mean"], mroas_lo=m["lo"], mroas_hi=m["hi"],
            saturation=s["mean"], half_life_weeks=h["mean"],
            prob_profitable=float((roas[:, c] > cfg.breakeven_roas).mean()),
        ))
    return pd.DataFrame(rows)


def response_curves(P: Params, grid_points: int = 40, level: float = 0.90, max_mult: float = 3.0):
    """Weekly steady-state incremental revenue vs weekly spend for each channel, with intervals.
    With normalised adstock, a constant weekly spend x has adstock x, so response = beta*T*tanh(lam*(x/s)/2)."""
    curves = {}
    for c, ch in enumerate(P.channels):
        grid = np.linspace(0, max(P.spend_mean[c], 1e-9) * max_mult, grid_points)
        u = grid / P.channel_scale[c]
        resp = P.beta[:, [c]] * P.target_scale * logistic_saturation(u[None, :], P.lam[:, [c]])  # [S, G]
        lo, hi = _ci(resp, 0, level)
        curves[ch] = dict(spend=grid, mean=resp.mean(axis=0), lo=lo, hi=hi)
    return curves


# ---------------------------------------------------------------------------------------------
# Revenue decomposition (from the model's own additive components)
# ---------------------------------------------------------------------------------------------
def decomposition(mmm, X: pd.DataFrame, level: float = 0.90) -> pd.DataFrame:
    """Share of total revenue explained by the non-media baseline and by each channel.
    Baseline = intercept + seasonality + trend + all controls (z-scored controls net to ~0 over time)."""
    post = _node(mmm.idata, "posterior")
    const = _node(mmm.idata, "constant_data")
    t_scale_val = const["target_scale"]
    if hasattr(t_scale_val, "values"):
        t_scale_val = t_scale_val.values
    ts = float(np.atleast_1d(t_scale_val)[0])

    non_media = _flat(post["intercept_contribution"]) * len(X)
    for k in ("yearly_seasonality_contribution", "control_contribution"):
        if k in post.data_vars:
            da = post[k]
            non_media = non_media + _flat(da.sum([d for d in da.dims if d not in ("chain", "draw")]))
    comps = {"baseline (non-media)": non_media * ts}
    cch = post["channel_contribution"].sum("date")
    arr = _flat(cch, "channel")
    for i, ch in enumerate([str(v) for v in cch["channel"].values]):
        comps[ch] = arr[:, i] * ts
    total = sum(comps.values())
    rows = []
    for k, v in comps.items():
        s, sh = summarise(v, level), summarise(v / total, level)
        rows.append(dict(component=k, revenue=s["mean"], share=sh["mean"], share_lo=sh["lo"], share_hi=sh["hi"]))
    return pd.DataFrame(rows)


def control_effects(mmm, level: float = 0.90) -> pd.DataFrame:
    """Effect on weekly revenue of +1 standard deviation in each control (dummies: effect of the event)."""
    post = _node(mmm.idata, "posterior")
    const = _node(mmm.idata, "constant_data")
    t_scale_val = const["target_scale"]
    if hasattr(t_scale_val, "values"):
        t_scale_val = t_scale_val.values
    ts = float(np.atleast_1d(t_scale_val)[0])

    g = post["gamma_control"]
    names = [str(v) for v in g["control"].values]
    arr = _flat(g, "control") * ts
    rows = []
    for i, n in enumerate(names):
        s = summarise(arr[:, i], level)
        rows.append(dict(control=n, effect=s["mean"], lo=s["lo"], hi=s["hi"],
                         prob_positive=float((arr[:, i] > 0).mean())))
    return pd.DataFrame(rows)
