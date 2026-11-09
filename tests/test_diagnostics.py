import numpy as np
import pandas as pd
from mmm.diagnostics import regression_metrics, scorecard, _seasonal_naive


def test_regression_metrics_perfect_and_bad():
    y = np.array([100.0, 110, 120])
    assert regression_metrics(y, y)["r2"] == 1.0
    assert regression_metrics(y, y * 1.1)["mape"] > 0.09


def test_seasonal_naive_uses_lag_52():
    y = np.arange(200, dtype=float)
    # With 200 train elements and a lag of 52, 
    # test index 0 (actual index 200) should look back to 200 - 52 = 148.
    # test index 1 (actual index 201) should look back to 201 - 52 = 149.
    np.testing.assert_allclose(_seasonal_naive(y, np.array([150, 151])), [148, 149])


def _inputs(rhat=1.002, r2=0.9):
    conv = dict(max_rhat=rhat, min_ess=2000, divergence_frac=0.0)
    fit = dict(r2=r2, coverage=0.9, durbin_watson=2.0, ljung_box_p=0.5)
    cv = pd.DataFrame(dict(mape=[0.05], naive_mape=[0.2], converged=[True]))
    aud = dict(vif={"a": 1.2}, obs_per_parameter=12)
    return conv, fit, cv, aud


def test_scorecard_verdicts():
    assert scorecard(*_inputs())["verdict"].startswith("DECISION-GRADE")
    assert scorecard(*_inputs(rhat=1.2))["verdict"].startswith("NOT RELIABLE")
    assert scorecard(*_inputs(r2=0.7))["verdict"].startswith("DIRECTIONAL")
