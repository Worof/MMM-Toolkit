"""Budget optimisation on posterior response curves, with business constraints and uncertainty.

Weekly steady-state view: each channel receives a constant weekly spend x_c; the incremental revenue is
    R_c(x_c) = beta_c * target_scale * tanh(lam_c * (x_c / channel_scale_c) / 2)
(valid because normalised adstock of a constant series is the constant itself). The optimiser maximises
the posterior-mean total response (optionally penalising its posterior std = risk aversion) subject to
    sum_c x_c = budget   and   min_mult * current_c <= x_c <= max_mult * current_c.
Bounds keep the plan inside the spend range the data can actually support."""
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .analysis import Params, logistic_saturation, summarise
from .config import CFG, MMMConfig


def response_draws(P: Params, x: np.ndarray) -> np.ndarray:
    """Incremental weekly revenue draws per channel for weekly spend vector x: [S, C]."""
    u = np.asarray(x, float) / P.channel_scale
    return P.beta * P.target_scale * logistic_saturation(u[None, :], P.lam)


def total_response(P: Params, x: np.ndarray) -> np.ndarray:
    return response_draws(P, x).sum(axis=1)                                    # [S]


def marginal_roas(P: Params, x: np.ndarray) -> np.ndarray:
    """Expected revenue from the next dollar in each channel at spend x: [C] (posterior mean)."""
    u = np.asarray(x, float) / P.channel_scale
    sech2 = 1.0 / np.cosh(P.lam * u[None, :] / 2.0) ** 2
    return (P.beta * P.target_scale * (P.lam / 2.0) * sech2 / P.channel_scale).mean(axis=0)


def evaluate_plan(P: Params, x: np.ndarray, level: float = 0.90) -> Dict[str, float]:
    return summarise(total_response(P, x), level)


def optimize_allocation(P: Params, total_budget: Optional[float] = None, min_mult: float = None,
                        max_mult: float = None, risk_aversion: float = None,
                        cfg: MMMConfig = CFG) -> dict:
    min_mult = cfg.min_spend_mult if min_mult is None else min_mult
    max_mult = cfg.max_spend_mult if max_mult is None else max_mult
    risk = cfg.risk_aversion if risk_aversion is None else risk_aversion
    cur = np.asarray(P.spend_mean, float)
    budget = float(total_budget if total_budget is not None else cur.sum())
    lo, hi = min_mult * cur, max_mult * cur
    if budget < lo.sum() - 1e-6 or budget > hi.sum() + 1e-6:
        raise ValueError(
            f"Budget {budget:,.0f} is outside the supported range [{lo.sum():,.0f}, {hi.sum():,.0f}] "
            f"given per-channel bounds ({min_mult:.0%}-{max_mult:.0%} of current spend). "
            "Extrapolating beyond observed spend would be unreliable.")

    base = max(float(total_response(P, cur).mean()), 1.0)

    def neg_objective(m):
        r = total_response(P, m * cur)
        return -(r.mean() - risk * r.std()) / base

    m0 = np.clip(np.full_like(cur, budget / cur.sum()), min_mult, max_mult)
    res = minimize(
        neg_objective, m0, method="SLSQP",
        bounds=[(min_mult, max_mult)] * len(cur),
        constraints=[{"type": "eq", "fun": lambda m: float(np.sum(m * cur) - budget)}],
        options={"ftol": 1e-9, "maxiter": 500},
    )
    x_opt = res.x * cur
    # numerical clean-up so the budget constraint holds exactly
    x_opt = np.clip(x_opt, lo, hi)
    x_opt = x_opt * (budget / x_opt.sum()) if x_opt.sum() > 0 else x_opt

    r_cur, r_opt = response_draws(P, cur), response_draws(P, x_opt)
    baseline_x = cur * budget / cur.sum()      # 'do nothing smart': scale today's mix to the new budget
    uplift = r_opt.sum(axis=1) - total_response(P, baseline_x)
    table = pd.DataFrame(dict(
        channel=P.channels, current=cur, recommended=x_opt,
        change_pct=np.where(cur > 0, x_opt / np.maximum(cur, 1e-9) - 1.0, 0.0),
        response_current=r_cur.mean(axis=0), response_recommended=r_opt.mean(axis=0),
        mroas_current=marginal_roas(P, cur), mroas_recommended=marginal_roas(P, x_opt),
        at_bound=np.isclose(x_opt, lo, rtol=1e-3) | np.isclose(x_opt, hi, rtol=1e-3),
    ))
    return dict(
        table=table, budget=budget, converged=bool(res.success),
        current=summarise(total_response(P, cur), cfg.interval),
        recommended=summarise(r_opt.sum(axis=1), cfg.interval),
        uplift=summarise(uplift, cfg.interval),
        prob_uplift_positive=float((uplift > 0).mean()),
        risk_aversion=risk,
    )


def budget_scenarios(P: Params, multipliers=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
                     cfg: MMMConfig = CFG) -> pd.DataFrame:
    """Optimal expected return at different total budgets: shows diminishing returns of the whole plan."""
    cur_total = float(P.spend_mean.sum())
    rows, prev = [], None
    for m in multipliers:
        try:
            r = optimize_allocation(P, cur_total * m, cfg=cfg)
        except ValueError:
            continue
        resp = r["recommended"]
        rows.append(dict(budget_mult=m, budget=cur_total * m, response=resp["mean"],
                         lo=resp["lo"], hi=resp["hi"],
                         marginal_return=(np.nan if prev is None else
                                          (resp["mean" ] - prev[1]) / max(cur_total * m - prev[0], 1e-9))))
        prev = (cur_total * m, resp["mean"])
    return pd.DataFrame(rows)
