import numpy as np
import pandas as pd
import pytest
from mmm.config import MMMConfig
from mmm.data import audit, fit_spec, make_design, time_series_splits


def _df(n=60):
    cfg = MMMConfig(spend_columns=["a_S", "b_S"], control_columns=["price"], event_column="events")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"DATE": pd.date_range("2022-01-03", periods=n, freq="W-MON"),
                       "revenue": rng.uniform(90, 110, n), "a_S": rng.uniform(0, 10, n),
                       "b_S": rng.uniform(0, 5, n), "price": rng.normal(1e6, 1e5, n),
                       "events": ["na"] * n})
    df.loc[40, "events"] = "launch"
    return df, cfg


def test_controls_are_standardised_with_train_stats_only():
    df, cfg = _df()
    train, test = df.iloc[:50], df.iloc[50:]
    spec = fit_spec(train, cfg)
    Xtr, _ = make_design(train, spec, cfg)
    Xte, _ = make_design(test, spec, cfg)
    assert abs(Xtr["price"].mean()) < 1e-9 and abs(Xtr["price"].std(ddof=0) - 1) < 1e-9
    assert abs(Xte["price"].mean()) > 1e-6                      # test is NOT re-centred (no leakage)
    assert list(Xte.columns) == list(Xtr.columns)


def test_unseen_event_in_holdout_is_zero_filled():
    df, cfg = _df()
    spec = fit_spec(df.iloc[:30], cfg)                           # 'launch' (row 40) not in train
    assert "event_launch" not in spec.controls
    X, _ = make_design(df.iloc[30:], spec, cfg)
    assert "event_launch" not in X.columns


def test_event_dummies_and_trend_present():
    df, cfg = _df()
    spec = fit_spec(df, cfg)
    X, _ = make_design(df, spec, cfg)
    assert X["event_launch"].sum() == 1 and "trend" in X.columns


def test_audit_flags_collinearity():
    df, cfg = _df()
    df["b_S"] = df["a_S"] * 2 + 1e-3 * np.arange(len(df))
    a = audit(df, cfg)
    assert a["high_correlation_pairs"] and max(a["vif"].values()) > 5


def test_splits_expanding_nonoverlapping():
    s = time_series_splits(200, 4, 20)
    for (tr, te), (tr2, te2) in zip(s, s[1:]):
        assert len(tr2) > len(tr) and te.max() < te2.min()
    assert all(tr.max() < te.min() for tr, te in s) and s[-1][1][-1] == 199
    with pytest.raises(ValueError):
        time_series_splits(50, 4, 20)
