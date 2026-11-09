"""Streamlit dashboard: findings, ROAS, what-if optimiser, response curves, model health.
Run `mmm-run` first (creates artifacts/), then: streamlit run app.py"""
import json
import os
import numpy as np
import pandas as pd
import streamlit as st

from mmm.analysis import Params, response_curves
from mmm.config import CFG
from mmm.optimize import budget_scenarios, evaluate_plan, optimize_allocation

st.set_page_config(page_title="Marketing Mix Model", layout="wide")
ART = CFG.artifacts_dir
if not os.path.exists(os.path.join(ART, "params.npz")):
    st.error(f"No model artifacts in '{ART}/'. Run `mmm-run` first.")
    st.stop()


@st.cache_data
def load():
    P = Params.load(os.path.join(ART, "params.npz"))
    with open(os.path.join(ART, "results.json")) as f:
        R = json.load(f)
    return P, R


P, R = load()
cur = CFG.currency
st.title("Marketing Mix Model: what works, and where to put the next dollar")
(st.success if R["reliable"] else st.error)(f"Model status: {R['verdict']}")

t1, t2, t3, t4, t5 = st.tabs(["Channel efficiency", "Budget optimiser", "Response curves", "Model health", "Report"])

with t1:
    roas = pd.DataFrame(R["roas"])
    c = st.columns(4)
    c[0].metric("In-sample R²", f"{R['fit']['r2']:.2f}")
    if R.get("cv"):
        cv = pd.DataFrame(R["cv"])
        c[1].metric("Out-of-sample MAPE", f"{cv['mape'].mean():.1%}", f"vs naive {cv['naive_mape'].mean():.1%}",
                    delta_color="off")
    base = [d for d in R["decomposition"] if d["component"].startswith("baseline")][0]
    c[2].metric("Revenue driven by media", f"{1 - base['share']:.0%}")
    best = roas.sort_values("mroas", ascending=False).iloc[0]
    c[3].metric("Best marginal ROAS", best["channel"], f"{best['mroas']:.1f}", delta_color="off")
    show = roas[["channel", "spend", "roas", "roas_lo", "roas_hi", "mroas", "saturation",
                 "half_life_weeks", "prob_profitable"]]
    st.dataframe(show.style.format({"spend": "{:,.0f}", "roas": "{:.2f}", "roas_lo": "{:.2f}", "roas_hi": "{:.2f}",
                                    "mroas": "{:.2f}", "saturation": "{:.0%}", "half_life_weeks": "{:.1f}",
                                    "prob_profitable": "{:.0%}"}), width="stretch", hide_index=True)
    st.caption("ROAS = incremental revenue / spend. Marginal ROAS = revenue from the next dollar at today's spend. "
               "lo/hi = 90% credible interval.")
    st.bar_chart(pd.DataFrame(R["decomposition"]).set_index("component")["revenue"])

with t2:
    total_now = float(P.spend_mean.sum())
    a, b, d = st.columns(3)
    budget = a.slider(f"Total weekly budget ({cur})", int(total_now * 0.5), int(total_now * 2.0), int(total_now),
                      step=max(1, int(total_now * 0.01)))
    mn, mx = b.slider("Per-channel bounds (x current)", 0.0, 3.0, (CFG.min_spend_mult, CFG.max_spend_mult), 0.1)
    risk = d.slider("Risk aversion", 0.0, 2.0, 0.0, 0.1,
                    help="0 maximises expected revenue; higher favours channels with tighter uncertainty.")
    try:
        res = optimize_allocation(P, budget, mn, mx, risk)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    t = res["table"]
    u = res["uplift"]
    k = st.columns(3)
    k[0].metric("Expected weekly gain vs proportional mix", f"{cur}{u['mean']:,.0f}",
                f"{cur}{u['lo']:,.0f} to {cur}{u['hi']:,.0f}", delta_color="off")
    k[1].metric("Annualised", f"{cur}{u['mean'] * 52:,.0f}")
    k[2].metric("P(beats current mix)", f"{res['prob_uplift_positive']:.0%}")
    st.bar_chart(t.set_index("channel")[["current", "recommended"]])
    st.dataframe(t[["channel", "current", "recommended", "change_pct", "mroas_current", "mroas_recommended"]]
                 .style.format({"current": "{:,.0f}", "recommended": "{:,.0f}", "change_pct": "{:+.0%}",
                                "mroas_current": "{:.2f}", "mroas_recommended": "{:.2f}"}),
                 width="stretch", hide_index=True)
    st.subheader("Your own plan")
    custom = np.array([st.number_input(ch, 0.0, float(P.spend_mean[i] * 5), float(round(P.spend_mean[i])),
                                       key=f"c{i}") for i, ch in enumerate(P.channels)])
    ev = evaluate_plan(P, custom)
    st.metric("Expected incremental weekly revenue", f"{cur}{ev['mean']:,.0f}",
              f"{cur}{ev['lo']:,.0f} to {cur}{ev['hi']:,.0f}", delta_color="off")
    sc = budget_scenarios(P)
    if len(sc):
        st.line_chart(sc.set_index("budget")[["lo", "response", "hi"]])
        st.caption("Flattening at high budgets partly reflects the per-channel spend caps.")

with t3:
    curves = response_curves(P)
    cols = st.columns(3)
    for i, ch in enumerate(P.channels):
        c = curves[ch]
        cols[i % 3].caption(ch)
        cols[i % 3].line_chart(pd.DataFrame({"lo": c["lo"], "mean": c["mean"], "hi": c["hi"]}, index=c["spend"]))

with t4:
    sc_df = pd.DataFrame(R["scorecard"]).astype({"value": str})
    st.dataframe(sc_df, width="stretch", hide_index=True)
    if R.get("cv"):
        st.subheader("Cross-validation")
        st.dataframe(pd.DataFrame(R["cv"]), width="stretch", hide_index=True)
    st.json(R["audit"], expanded=False)

with t5:
    path = os.path.join(ART, "mmm_report.html")
    if os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button("Download executive report (HTML)", f, "mmm_report.html", "text/html")
