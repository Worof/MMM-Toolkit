# Model card: MMM Toolkit

**Model:** Bayesian MMM (pymc-marketing 1.x). Geometric adstock (l_max = 8, normalised), logistic saturation,
Fourier seasonality (order 2), linear trend, z-scored controls, event dummies. Spend-aware priors on effect size.

**Validation (all automatic, see the scorecard in the report):** R-hat, ESS, divergences on sampled parameters;
in-sample R^2 and interval coverage; Durbin-Watson and Ljung-Box on residuals; expanding-window time-series CV
versus a seasonal-naive benchmark; prior sensitivity; recovery of known ROAS on synthetic data (tests/test_recovery.py).

**Decision rule:** DECISION-GRADE (all checks pass) / DIRECTIONAL (warnings: validate large moves with an experiment) /
NOT RELIABLE (convergence or fit failure: do not act).

**Optimiser:** steady-state weekly view, per-channel bounds (default 50% to 200% of current spend), optional risk aversion.
Budgets outside the supported range are rejected rather than extrapolated.

**Limits**
- Observational data: effects are associations unless calibrated with experiments (lift tests).
- ROAS is revenue per unit of spend; profitability depends on margin (config: breakeven_roas).
- Intervals capture parameter uncertainty, not model-misspecification risk.
- The bundled dataset is Meta Robyn's simulated data, not real business results.
