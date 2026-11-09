"""End-to-end run: audit -> validate (CV) -> fit -> diagnose -> insights -> optimise -> report."""
import argparse
import json
import os
from typing import Optional
import numpy as np
import pandas as pd

from .analysis import (Params, control_effects, decomposition, extract_params, response_curves, roas_table)
from .config import CFG, MMMConfig
from .data import audit, fit_spec, load_raw, make_design, print_audit
from .diagnostics import convergence, cross_validate, in_sample_fit, prior_sensitivity, scorecard
from .model import add_lift_tests, build_mmm, fit_mmm
from .optimize import budget_scenarios, optimize_allocation
from .report import build_report


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(type(o))


def run(data_path: Optional[str] = None, out_dir: Optional[str] = None, lift_tests: Optional[str] = None,
        skip_cv: bool = False, robustness: bool = False, mlflow_log: bool = False,
        cfg: MMMConfig = CFG, verbose: bool = True) -> dict:
    out_dir = out_dir or cfg.artifacts_dir
    os.makedirs(out_dir, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    say("1/6 Data audit")
    df = load_raw(data_path, cfg)
    aud = audit(df, cfg)
    if verbose:
        print_audit(aud)

    cv = None
    if not skip_cv:
        say(f"2/6 Time-series cross-validation ({cfg.cv_folds} folds x {cfg.cv_test_size} weeks)")
        cv = cross_validate(df, cfg, verbose=verbose)

    say("3/6 Final fit on all data")
    spec = fit_spec(df, cfg)
    X, y = make_design(df, spec, cfg)
    mmm = build_mmm(X, cfg)
    if lift_tests:
        say(f"    calibrating with experiments from {lift_tests}")
        mmm.build_model(X, y)
        add_lift_tests(mmm, pd.read_csv(lift_tests))
    mmm = fit_mmm(mmm, X, y, cfg)

    say("4/6 Diagnostics")
    conv = convergence(mmm, cfg)
    fit = in_sample_fit(mmm, X, y, cfg)
    sc = scorecard(conv, fit, cv, aud, cfg)
    say(f"    verdict: {sc['verdict']}")

    say("5/6 Insights and optimisation")
    P = extract_params(mmm, X, cfg)
    spend = X[cfg.spend_columns].values
    R = dict(
        audit=aud, cv=cv, convergence=conv, fit=fit, scorecard=sc, params=P,
        roas=roas_table(P, spend, cfg.interval, cfg),
        decomposition=decomposition(mmm, X, cfg.interval),
        control_effects=control_effects(mmm, cfg.interval),
        curves=response_curves(P, level=cfg.interval),
        optimization=optimize_allocation(P, cfg=cfg),
        scenarios=budget_scenarios(P, cfg=cfg),
        sensitivity=prior_sensitivity(df, cfg) if robustness else None,
    )

    say("6/6 Saving artifacts")
    P.save(os.path.join(out_dir, "params.npz"))
    summary = dict(
        verdict=sc["verdict"], reliable=sc["reliable"], scorecard=sc["checks"], audit=aud,
        convergence=conv, fit={k: v for k, v in fit.items() if isinstance(v, float)},
        roas=R["roas"].to_dict("records"), decomposition=R["decomposition"].to_dict("records"),
        control_effects=R["control_effects"].to_dict("records"),
        optimization={k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v)
                      for k, v in R["optimization"].items()},
        scenarios=R["scenarios"].to_dict("records"),
        cv=None if cv is None else cv.to_dict("records"),
    )
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_jsonable)
    report_path = build_report(R, os.path.join(out_dir, "mmm_report.html"), cfg)
    say(f"    report: {report_path}")

    if mlflow_log:
        import mlflow
        mlflow.set_experiment("mmm-toolkit")
        with mlflow.start_run():
            mlflow.log_params({k: v for k, v in cfg.model_dump().items() if np.isscalar(v)})
            mlflow.log_metrics({"r2": fit["r2"], "mape_in_sample": fit["mape"], "max_rhat": conv["max_rhat"],
                                "min_ess": conv["min_ess"], "divergences": conv["divergences"],
                                **({"cv_mape": float(cv["mape"].mean())} if cv is not None else {})})
            for fn in ("results.json", "mmm_report.html", "params.npz"):
                mlflow.log_artifact(os.path.join(out_dir, fn))
    R["mmm"] = mmm
    return R


def main():
    p = argparse.ArgumentParser(description="Run the full Marketing Mix Modeling pipeline")
    p.add_argument("--data", default=None, help="CSV path (default: config data_path)")
    p.add_argument("--out", default=None, help="Output directory (default: artifacts/)")
    p.add_argument("--lift-tests", default=None, help="CSV of experiments: channel,x,delta_x,delta_y,sigma")
    p.add_argument("--skip-cv", action="store_true", help="Skip cross-validation (faster, less evidence)")
    p.add_argument("--robustness", action="store_true", help="Also run prior-sensitivity refits")
    p.add_argument("--mlflow", action="store_true", help="Log run to MLflow")
    a = p.parse_args()
    run(a.data, a.out, a.lift_tests, a.skip_cv, a.robustness, a.mlflow)


if __name__ == "__main__":
    main()
