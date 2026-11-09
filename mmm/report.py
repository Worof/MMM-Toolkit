"""Self-contained executive HTML report: findings first, evidence second, caveats explicit."""
import base64
import html
import io
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import CFG, MMMConfig

PALETTE = ["#2563eb", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#14b8a6", "#6b7280"]


def _money(v: float, cur: str = "$") -> str:
    a = abs(v)
    s = f"{a / 1e9:.2f}B" if a >= 1e9 else f"{a / 1e6:.2f}M" if a >= 1e6 else f"{a / 1e3:.1f}K" if a >= 1e3 else f"{a:,.0f}"
    return f"{'-' if v < 0 else ''}{cur}{s}"


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(fig, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{_png(fig)}"/>'


# ---------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------
def fig_fit(fit: dict, cfg: MMMConfig):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    d = pd.to_datetime(fit["dates"])
    ax.fill_between(d, fit["lo"], fit["hi"], color="#93c5fd", alpha=0.45, label=f"{cfg.interval:.0%} predictive interval")
    ax.plot(d, fit["pred"], color="#2563eb", lw=1.5, label="Model")
    ax.plot(d, fit["actual"], color="#111827", lw=1, label="Actual")
    ax.set_title(f"Model fit (R² = {fit['r2']:.2f}, MAPE = {fit['mape']:.1%})", loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: _money(v, ""))
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_roas(roas: pd.DataFrame, cfg: MMMConfig):
    r = roas.sort_values("roas")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    y = np.arange(len(r))
    ax.barh(y, r["roas"], color="#2563eb", alpha=0.85)
    ax.errorbar(r["roas"], y, xerr=[r["roas"] - r["roas_lo"], r["roas_hi"] - r["roas"]], fmt="none",
                ecolor="#111827", capsize=3, lw=1)
    ax.axvline(cfg.breakeven_roas, color="#ef4444", ls="--", lw=1)
    ax.text(cfg.breakeven_roas, len(r) - 0.4, " break-even", color="#ef4444", fontsize=8, va="bottom")
    ax.set_yticks(y, r["channel"])
    ax.set_title(f"ROAS with {cfg.interval:.0%} credible interval", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    spend_sh = r["spend"] / r["spend"].sum()
    rev_sh = r["incremental_revenue"] / r["incremental_revenue"].sum()
    h = 0.38
    ax.barh(y + h / 2, spend_sh, h, label="Share of spend", color="#9ca3af")
    ax.barh(y - h / 2, rev_sh, h, label="Share of media-driven revenue", color="#10b981")
    ax.set_yticks(y, [""] * len(y))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Where money goes vs where revenue comes from", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def fig_curves(curves: dict, current: np.ndarray, recommended: np.ndarray, channels: List[str], cfg: MMMConfig):
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.0), sharey=False)
    axes = np.atleast_1d(axes)
    for i, (ax, ch) in enumerate(zip(axes, channels)):
        c = curves[ch]
        ax.fill_between(c["spend"], c["lo"], c["hi"], color=PALETTE[i % 7], alpha=0.2)
        ax.plot(c["spend"], c["mean"], color=PALETTE[i % 7], lw=2)
        ax.axvline(current[i], color="#111827", ls=":", lw=1.2)
        ax.axvline(recommended[i], color="#10b981", ls="--", lw=1.2)
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel("weekly spend", fontsize=8)
        ax.xaxis.set_major_formatter(lambda v, _: _money(v, ""))
        ax.yaxis.set_major_formatter(lambda v, _: _money(v, ""))
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("incremental weekly revenue", fontsize=8)
    fig.suptitle("Response curves (diminishing returns). Dotted = current, dashed green = recommended",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return fig


def fig_allocation(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["current"], 0.4, label="Current", color="#9ca3af")
    ax.bar(x + 0.2, table["recommended"], 0.4, label="Recommended", color="#10b981")
    ax.set_xticks(x, table["channel"])
    ax.yaxis.set_major_formatter(lambda v, _: _money(v, ""))
    ax.legend(frameon=False)
    ax.set_title("Weekly budget: current vs recommended", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_scenarios(sc: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.fill_between(sc["budget"], sc["lo"], sc["hi"], color="#93c5fd", alpha=0.4)
    ax.plot(sc["budget"], sc["response"], marker="o", color="#2563eb")
    ax.xaxis.set_major_formatter(lambda v, _: _money(v, ""))
    ax.yaxis.set_major_formatter(lambda v, _: _money(v, ""))
    ax.set_xlabel("total weekly budget (optimally allocated)")
    ax.set_ylabel("incremental weekly revenue")
    ax.set_title("What if the total budget changes?", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# ---------------------------------------------------------------------------------------------
# Narrative
# ---------------------------------------------------------------------------------------------
def executive_summary(R: Dict, cfg: MMMConfig = CFG) -> List[str]:
    cur = cfg.currency
    sc, roas, dec, opt = R["scorecard"], R["roas"], R["decomposition"], R["optimization"]
    pts = []
    cvm = R.get("cv")
    perf = f"in-sample R² {R['fit']['r2']:.2f}"
    if cvm is not None and len(cvm):
        perf += f"; out-of-sample error {cvm['mape'].mean():.1%} vs {cvm['naive_mape'].mean():.1%} for a seasonal-naive forecast"
    pts.append(f"<b>Model reliability: {html.escape(sc['verdict'])}.</b> {perf}. "
               f"{sc['n_pass']} checks passed, {sc['n_warn']} warnings, {sc['n_fail']} failures.")

    base = dec.loc[dec["component"].str.startswith("baseline")].iloc[0]
    pts.append(f"<b>Marketing drives about {1 - base['share']:.0%} of revenue</b> "
               f"({1 - base['share_hi']:.0%} to {1 - base['share_lo']:.0%} at {cfg.interval:.0%} confidence); "
               f"the rest is baseline demand, seasonality and non-media factors.")

    lose = roas[roas["prob_profitable"] < 0.5].sort_values("roas")
    for _, r in lose.iterrows():
        sh = r["spend"] / roas["spend"].sum()
        pts.append(f"<b>{r['channel']} is probably not paying back:</b> ROAS {r['roas']:.2f} "
                   f"({r['roas_lo']:.2f} to {r['roas_hi']:.2f}), only a {r['prob_profitable']:.0%} chance of beating "
                   f"break-even, while taking {sh:.0%} of spend.")
    top = roas.sort_values("mroas", ascending=False).iloc[0]
    pts.append(f"<b>Best place for the next dollar: {top['channel']}</b> (marginal ROAS {top['mroas']:.1f}, "
               f"range {top['mroas_lo']:.1f} to {top['mroas_hi']:.1f}).")

    t = opt["table"]
    ups = t[t["change_pct"] > 0.05].sort_values("change_pct", ascending=False)["channel"].tolist()
    downs = t[t["change_pct"] < -0.05].sort_values("change_pct")["channel"].tolist()
    if ups or downs:
        u = opt["uplift"]
        pts.append(f"<b>Recommendation:</b> at the same total budget, shift money from "
                   f"{', '.join(downs) or 'no channel'} to {', '.join(ups) or 'no channel'}. "
                   f"Expected gain {_money(u['mean'], cur)} per week "
                   f"({_money(u['lo'], cur)} to {_money(u['hi'], cur)}), about {_money(u['mean'] * 52, cur)} a year; "
                   f"it beats the current mix in {opt['prob_uplift_positive']:.0%} of plausible worlds.")
    wide = roas.assign(rel=(roas["roas_hi"] - roas["roas_lo"]) / roas["roas"].clip(lower=1e-9)) \
        .sort_values("rel", ascending=False).iloc[0]
    pts.append(f"<b>Biggest uncertainty:</b> {wide['channel']} (ROAS {wide['roas_lo']:.1f} to {wide['roas_hi']:.1f}). "
               f"Phase large shifts and run a geo-holdout test there; feed the result back as a lift test to tighten the model.")
    return pts


CAVEATS = [
    "Observational data: effects are estimated associations. Experiments (geo-lift, holdouts) are the gold standard; "
    "the toolkit can ingest them as calibration (<code>--lift-tests</code>).",
    "ROAS here is <i>revenue</i> per currency unit of spend. Profitability depends on margin; the break-even line is configurable.",
    "Recommendations use steady-state weekly spend and are limited to the range the data supports "
    f"({CFG.min_spend_mult:.0%} to {CFG.max_spend_mult:.0%} of today's spend per channel). Phase big moves.",
    "Curves assume geometric carry-over and logistic saturation; channels with little spend variation "
    "(flighted or tiny) have wide intervals. Wide intervals are information, not noise.",
    "Intervals reflect parameter uncertainty within the model, not model-misspecification risk. "
    "The prior-sensitivity table shows how much conclusions move under alternative priors.",
]


def _table(df: pd.DataFrame, fmt: Dict[str, str] = None, cls: str = "") -> str:
    fmt = fmt or {}
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            tds.append(f"<td>{fmt[c](v) if c in fmt else html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def build_report(R: Dict, path: str, cfg: MMMConfig = CFG) -> str:
    cur = cfg.currency
    roas, opt, P = R["roas"], R["optimization"], R["params"]
    t = opt["table"]
    status_cls = {"pass": "ok", "warn": "warn", "fail": "bad"}

    sc_rows = "".join(
        f'<tr><td>{html.escape(c["check"])}</td><td>{html.escape(str(c["value"]))}</td>'
        f'<td>{html.escape(c["rule"])}</td><td class="{status_cls[c["status"]]}">{c["status"].upper()}</td></tr>'
        for c in R["scorecard"]["checks"])

    roas_view = roas[["channel", "spend", "roas", "roas_lo", "roas_hi", "mroas", "saturation", "half_life_weeks",
                      "prob_profitable"]].copy()
    roas_view.columns = ["Channel", "Spend (history)", "ROAS", "lo", "hi", "Marginal ROAS", "Saturation",
                         "Carry-over half-life (wk)", "P(profitable)"]
    roas_html = _table(roas_view, {
        "Spend (history)": lambda v: _money(v, cur), "ROAS": lambda v: f"{v:.2f}", "lo": lambda v: f"{v:.2f}",
        "hi": lambda v: f"{v:.2f}", "Marginal ROAS": lambda v: f"{v:.2f}", "Saturation": lambda v: f"{v:.0%}",
        "Carry-over half-life (wk)": lambda v: f"{v:.1f}", "P(profitable)": lambda v: f"{v:.0%}"})

    opt_view = t[["channel", "current", "recommended", "change_pct", "mroas_current", "mroas_recommended"]].copy()
    opt_view.columns = ["Channel", "Current / wk", "Recommended / wk", "Change", "mROAS now", "mROAS after"]
    opt_html = _table(opt_view, {
        "Current / wk": lambda v: _money(v, cur), "Recommended / wk": lambda v: _money(v, cur),
        "Change": lambda v: f"{v:+.0%}", "mROAS now": lambda v: f"{v:.2f}", "mROAS after": lambda v: f"{v:.2f}"})

    ctrl = R["control_effects"].copy()
    ctrl.columns = ["Driver", "Effect on weekly revenue", "lo", "hi", "P(>0)"]
    ctrl_html = _table(ctrl, {"Effect on weekly revenue": lambda v: _money(v, cur), "lo": lambda v: _money(v, cur),
                              "hi": lambda v: _money(v, cur), "P(>0)": lambda v: f"{v:.0%}"})

    cv_html = ""
    if R.get("cv") is not None and len(R["cv"]):
        cvv = R["cv"][["fold", "train_weeks", "test_weeks", "mape", "naive_mape", "coverage", "max_rhat"]].copy()
        cvv.columns = ["Fold", "Train wk", "Test wk", "MAPE", "Seasonal-naive MAPE", "Interval coverage", "R-hat"]
        cv_html = "<h3>Out-of-sample validation (expanding window)</h3>" + _table(cvv, {
            "MAPE": lambda v: f"{v:.1%}", "Seasonal-naive MAPE": lambda v: f"{v:.1%}",
            "Interval coverage": lambda v: f"{v:.0%}", "R-hat": lambda v: f"{v:.3f}"})

    sens_html = ""
    if R.get("sensitivity") is not None:
        s = R["sensitivity"].reset_index().rename(columns={"channel": "Channel"})
        fm = {c: (lambda v: f"{v:.2f}") for c in s.columns if c.startswith("beta_prior")}
        sens_html = ("<h3>Prior sensitivity (ROAS under alternative priors)</h3>"
                     + _table(s, fm) + "<p class='muted'>Stable ranking across priors means the conclusions are driven by the data.</p>")

    figs = dict(
        fit=_img(fig_fit(R["fit"], cfg), "model fit"),
        roas=_img(fig_roas(roas, cfg), "ROAS"),
        curves=_img(fig_curves(R["curves"], t["current"].values, t["recommended"].values, P.channels, cfg),
                    "response curves"),
        alloc=_img(fig_allocation(t), "allocation"),
        scen=_img(fig_scenarios(R["scenarios"]), "budget scenarios") if len(R["scenarios"]) else "",
    )
    bullets = "".join(f"<li>{p}</li>" for p in executive_summary(R, cfg))
    caveats = "".join(f"<li>{c}</li>" for c in CAVEATS)
    a = R["audit"]
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marketing Mix Model Report</title>
<style>
:root{{--bg:#fff;--fg:#111827;--mut:#6b7280;--line:#e5e7eb;--card:#f9fafb}}
@media (prefers-color-scheme: dark){{:root{{--bg:#0f172a;--fg:#e5e7eb;--mut:#94a3b8;--line:#334155;--card:#1e293b}}}}
body{{font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);max-width:980px;margin:0 auto;padding:24px}}
h1{{margin:0 0 4px}} h2{{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.muted{{color:var(--mut)}} .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 18px}}
table{{border-collapse:collapse;width:100%;margin:10px 0;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right}}
th:first-child,td:first-child{{text-align:left}} img{{max-width:100%;background:#fff;border-radius:8px}}
.ok{{color:#059669;font-weight:600}} .warn{{color:#d97706;font-weight:600}} .bad{{color:#dc2626;font-weight:600}}
li{{margin:6px 0}} code{{background:var(--card);padding:1px 5px;border-radius:4px}}
</style></head><body>
<h1>Marketing Mix Model: findings and budget recommendation</h1>
<p class="muted">Data: {a['start']} to {a['end']} ({a['rows']} weeks) | Bayesian MMM with adstock + saturation | all intervals {cfg.interval:.0%} credible</p>
<h2>Executive summary</h2><div class="card"><ul>{bullets}</ul></div>
<h2>1. Recommended budget allocation</h2>
{figs['alloc']}{opt_html}
<p class="muted">Constraint: each channel stays within {cfg.min_spend_mult:.0%} to {cfg.max_spend_mult:.0%} of its current weekly spend.
Expected weekly revenue: current {_money(opt['current']['mean'], cur)} incremental vs recommended {_money(opt['recommended']['mean'], cur)}.</p>
{figs['curves']}{figs['scen']}
<p class="muted">Flattening at higher budgets partly reflects the per-channel spend caps (extra money is forced into channels already at their limit), not only diminishing returns.</p>
<h2>2. Channel efficiency</h2>{figs['roas']}{roas_html}
<p class="muted">ROAS = incremental revenue / spend over the period. Marginal ROAS = revenue from the next dollar at today's spend. Saturation = share of a channel's maximum effect already captured.</p>
<h2>3. Non-media drivers</h2>{ctrl_html}
<h2>4. Can we trust it?</h2>{figs['fit']}
<div class="card"><b>{html.escape(R['scorecard']['verdict'])}</b></div>
<table><thead><tr><th>Check</th><th>Value</th><th>Rule</th><th>Status</th></tr></thead><tbody>{sc_rows}</tbody></table>
{cv_html}{sens_html}
<h2>5. Assumptions and caveats</h2><ul>{caveats}</ul>
<h2>6. Data audit</h2>
<p class="muted">Weekly grid regular: {a['weekly_regular']} | outlier weeks: {', '.join(a['outlier_weeks']) or 'none'} |
high-correlation pairs: {a['high_correlation_pairs'] or 'none'} | observations per parameter: {a['obs_per_parameter']}</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
