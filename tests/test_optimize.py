import numpy as np
import pytest
from mmm.config import MMMConfig
from mmm.optimize import budget_scenarios, optimize_allocation, total_response


def test_constraints_hold(params):
    cfg = MMMConfig()
    r = optimize_allocation(params, cfg=cfg)
    t = r["table"]
    assert abs(t["recommended"].sum() - r["budget"]) < 1e-6 * r["budget"]
    assert (t["recommended"] >= cfg.min_spend_mult * t["current"] - 1e-6).all()
    assert (t["recommended"] <= cfg.max_spend_mult * t["current"] + 1e-6).all()


def test_optimum_beats_current_and_random_feasible_plans(params):
    r = optimize_allocation(params)
    cur = params.spend_mean
    best_obj = total_response(params, r["table"]["recommended"].values).mean()
    assert best_obj >= total_response(params, cur).mean() - 1e-6
    rng = np.random.default_rng(1)
    for _ in range(500):
        x = rng.uniform(0.5, 2.0, 3) * cur
        x = x * r["budget"] / x.sum()
        if (x < 0.5 * cur).any() or (x > 2.0 * cur).any():
            continue
        assert total_response(params, x).mean() <= best_obj + 1e-6


def test_rejects_unsupported_budget(params):
    with pytest.raises(ValueError):
        optimize_allocation(params, total_budget=params.spend_mean.sum() * 10)
    with pytest.raises(ValueError):
        optimize_allocation(params, total_budget=params.spend_mean.sum() * 0.1)


def test_more_budget_never_hurts(params):
    sc = budget_scenarios(params)
    assert (np.diff(sc["response"]) >= -1e-6).all()


def test_uplift_interval_ordered(params):
    r = optimize_allocation(params)
    assert r["uplift"]["lo"] <= r["uplift"]["mean"] <= r["uplift"]["hi"]
    assert 0.0 <= r["prob_uplift_positive"] <= 1.0
