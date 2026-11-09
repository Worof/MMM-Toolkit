import numpy as np
from mmm.analysis import (Params, channel_contributions, geometric_adstock, half_life,
                          logistic_saturation, roas_table, response_curves)
from mmm.optimize import marginal_roas, response_draws


def test_adstock_normalised_constant_series_is_identity_after_warmup():
    out = geometric_adstock(np.ones(30), np.array([0.5]), l_max=8)[0]
    np.testing.assert_allclose(out[8:], 1.0)
    assert out[0] < 1.0  # warm-up (zero-padded history)


def test_adstock_impulse_decays_geometrically():
    x = np.zeros(10); x[0] = 1.0
    out = geometric_adstock(x, np.array([0.5]), l_max=4)[0]
    w = 0.5 ** np.arange(4); w /= w.sum()
    np.testing.assert_allclose(out[:4], w)
    assert out[4:].sum() == 0


def test_saturation_matches_logistic_definition_and_is_bounded():
    u = np.linspace(0, 5, 50)
    lam = 2.3
    ref = (1 - np.exp(-lam * u)) / (1 + np.exp(-lam * u))
    np.testing.assert_allclose(logistic_saturation(u, lam), ref)
    assert logistic_saturation(u, lam).max() < 1


def test_half_life_known_value():
    assert abs(half_life(np.array([0.5]))[0] - 1.0) < 1e-9


def test_params_roundtrip(tmp_path, params):
    p = tmp_path / "p.npz"
    params.save(str(p))
    q = Params.load(str(p))
    np.testing.assert_allclose(q.beta, params.beta)
    assert q.channels == params.channels and q.l_max == params.l_max


def test_marginal_roas_matches_numerical_derivative(params):
    x = params.spend_mean
    h = 1e-3
    for c in range(3):
        up, dn = x.copy(), x.copy()
        up[c] += h; dn[c] -= h
        num = (response_draws(params, up)[:, c] - response_draws(params, dn)[:, c]).mean() / (2 * h)
        assert abs(num - marginal_roas(params, x)[c]) < 1e-4 * max(abs(num), 1)


def test_roas_table_consistency(params):
    spend = np.tile(params.spend_mean, (104, 1))
    t = roas_table(params, spend)
    assert (t["roas_lo"] <= t["roas"]).all() and (t["roas"] <= t["roas_hi"]).all()
    contrib = channel_contributions(params, spend)
    np.testing.assert_allclose(t["incremental_revenue"], contrib.sum(axis=1).mean(axis=0))
    assert ((t["saturation"] > 0) & (t["saturation"] < 1)).all()


def test_response_curves_are_monotone_and_concave(params):
    for ch, c in response_curves(params).items():
        assert (np.diff(c["mean"]) >= -1e-9).all()
        assert (np.diff(c["mean"], 2) <= 1e-6 * c["mean"].max()).all()
        assert (c["lo"] <= c["mean"] + 1e-9).all() and (c["mean"] <= c["hi"] + 1e-9).all()
