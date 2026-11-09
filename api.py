"""REST API over the fitted model. Run `mmm-run` first, then: uvicorn api:app"""
import json
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mmm.analysis import Params
from mmm.config import CFG
from mmm.optimize import optimize_allocation

app = FastAPI(title="MMM Toolkit API", version="1.0.0")


def _load():
    p = os.path.join(CFG.artifacts_dir, "params.npz")
    if not os.path.exists(p):
        raise HTTPException(503, "Model artifacts not found. Run `mmm-run` first.")
    return Params.load(p)


class OptimizeRequest(BaseModel):
    total_budget: Optional[float] = Field(None, gt=0, description="Weekly budget; default = current total")
    min_mult: float = Field(CFG.min_spend_mult, ge=0)
    max_mult: float = Field(CFG.max_spend_mult, gt=0)
    risk_aversion: float = Field(0.0, ge=0)


@app.get("/health")
def health():
    return {"status": "ok", "artifacts": os.path.exists(os.path.join(CFG.artifacts_dir, "params.npz"))}


@app.get("/summary")
def summary():
    p = os.path.join(CFG.artifacts_dir, "results.json")
    if not os.path.exists(p):
        raise HTTPException(503, "Model artifacts not found. Run `mmm-run` first.")
    with open(p) as f:
        R = json.load(f)
    return {"verdict": R["verdict"], "reliable": R["reliable"], "roas": R["roas"], "fit": R["fit"]}


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    P = _load()
    try:
        r = optimize_allocation(P, req.total_budget, req.min_mult, req.max_mult, req.risk_aversion)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"budget": r["budget"], "allocation": r["table"].to_dict("records"), "uplift": r["uplift"],
            "prob_uplift_positive": r["prob_uplift_positive"], "converged": r["converged"]}
