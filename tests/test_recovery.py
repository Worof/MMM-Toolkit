"""Slow end-to-end validation on synthetic data with KNOWN truth (run: pytest -m slow)."""
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_pipeline_recovers_true_roas_and_matches_library_math():
    from mmm.analysis import channel_contributions, extract_params, roas_table
    from mmm.data import fit_spec, make_design
    from mmm.diagnostics import convergence
    from mmm.model import build_mmm, fit_mmm
    from mmm.simulate import simulate

    df, truth, cfg = simulate()
    spec = fit_spec(df, cfg)
    X, y = make_design(df, spec, cfg)
    mmm = fit_mmm(build_mmm(X, cfg), X, y, cfg)
    assert convergence(mmm, cfg)["max_rhat"] < 1.02

    P = extract_params(mmm, X, cfg)
    spend = X[cfg.spend_columns].values

    # 1) our closed-form contributions equal the library's own channel_contribution
    post = mmm.idata["posterior"]
    lib = post["channel_contribution"].stack(s=("chain", "draw")).transpose("s", "date", "channel").values
    np.testing.assert_allclose(channel_contributions(P, spend), lib * P.target_scale, rtol=1e-6, atol=1e-6)

    # 2) estimated ROAS is close to the simulated truth and ranks channels correctly
    r = roas_table(P, spend, 0.9, cfg).merge(truth, on="channel")
    rel_err = (r["roas"] - r["true_roas"]).abs() / r["true_roas"]
    assert (rel_err < 0.4).all(), r[["channel", "roas", "true_roas"]]
    assert r.loc[r["roas"].idxmin(), "channel"] == r.loc[r["true_roas"].idxmin(), "channel"]
    covered = ((r["true_roas"] >= r["roas_lo"]) & (r["true_roas"] <= r["roas_hi"])).sum()
    assert covered >= 2
