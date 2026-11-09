# MMM Toolkit

Bayesian Marketing Mix Modeling (pymc-marketing): geometric adstock + logistic saturation, spend-aware priors,
time-series cross-validation, ROAS / marginal ROAS with credible intervals, constrained budget optimisation,
experiment calibration, executive HTML report, dashboard and API.

## Quick start
```bash
pip install -e ".[dev,api,dashboard,tracking]"
python -m pytest                       # unit tests + synthetic-truth recovery test
mmm-run --robustness                   # audit, CV, fit, report -> artifacts/
streamlit run app.py                   # dashboard
uvicorn api:app                        # REST API
mmm-run --lift-tests data/lift_tests.example.csv   # calibrate with experiment results
```

## Method in one paragraph
revenue_t = baseline + seasonality + trend + controls + sum_c beta_c * tanh(lam_c * adstock(spend_c,t) / 2) * scale.
Priors: beta_c ~ HalfNormal(0.5 * n_channels * spend_share_c), adstock alpha ~ Beta(1,3), lam ~ Gamma(3,1).
Controls are z-scored with train statistics only. ROAS = incremental revenue / spend. Marginal ROAS = revenue from the
next dollar. The optimiser maximises expected weekly response under per-channel bounds and reports the posterior
probability that the new mix beats the current one.

See MODEL_CARD.md for assumptions and limits.
