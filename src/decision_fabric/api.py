"""HTTP service — drop the fabric in front of an existing LLM gateway.

    pip install -r requirements-api.txt
    PYTHONPATH=src uvicorn decision_fabric.api:app --reload

POST /route   decide and execute
POST /plan    decide only (no model call) — use this to A/B against your
              current routing before you let it touch production traffic
GET  /savings cumulative spend vs the no-router baseline
GET  /graph   the fabric's current state, including learned drift
"""
from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .explain import render
from .ontology import EdgeType
from .router import Router

app = FastAPI(title="Decision Fabric", version="0.1.0")

_router: Optional[Router] = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router(
            db_path=os.environ.get("DECISION_FABRIC_DB", "./decision_fabric.db"),
            dry_run=None,  # live if credentials resolve, simulated otherwise
        )
    return _router


class RouteRequest(BaseModel):
    query: str
    policy: Optional[str] = Field(None, description="economy | balanced | quality | critical")
    latency_slo: str = "interactive"
    domain: Optional[str] = None
    task_type: Optional[str] = None
    context_tokens: int = 0
    stable_context_tokens: int = 0
    zero_data_retention: bool = False
    learn: bool = True
    explain: bool = False


@app.post("/route")
def route(req: RouteRequest) -> dict[str, Any]:
    r = get_router()
    try:
        d = r.route(
            req.query, execute=True, policy=req.policy, latency_slo=req.latency_slo,
            domain=req.domain, task_type=req.task_type,
            context_tokens=req.context_tokens,
            stable_context_tokens=req.stable_context_tokens,
            zero_data_retention=req.zero_data_retention, learn=req.learn,
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    out = d.to_dict()
    out["answer"] = d.answer
    out["request"] = d.selection.first_plan.to_request_kwargs(
        [{"role": "user", "content": "<query>"}]
    )
    if req.explain:
        out["report"] = render(d, show_trace=True)
    return out


@app.post("/plan")
def plan(req: RouteRequest) -> dict[str, Any]:
    """Shadow mode: what *would* this have been routed to, and for how much."""
    r = get_router()
    d = r.route(
        req.query, execute=False, policy=req.policy, latency_slo=req.latency_slo,
        domain=req.domain, task_type=req.task_type,
        context_tokens=req.context_tokens,
        stable_context_tokens=req.stable_context_tokens,
        zero_data_retention=req.zero_data_retention, learn=False,
    )
    out = d.to_dict()
    out["request"] = d.selection.first_plan.to_request_kwargs(
        [{"role": "user", "content": "<query>"}]
    )
    out["candidates"] = [
        {"model": c.model.id, "expected_quality": c.expected_quality,
         "est_cost_usd": round(c.est_cost_usd, 6), "eligible": c.eligible,
         "hard_ok": c.hard_ok, "reasons": c.reasons}
        for c in d.selection.all_candidates
    ]
    return out


@app.get("/savings")
def savings() -> dict[str, Any]:
    return get_router().telemetry.savings_report()


@app.get("/graph")
def graph() -> dict[str, Any]:
    r = get_router()
    drift = []
    for m in r.kg.models():
        for cap in m.provides:
            e = r.kg.edge(m.id, f"cap:{cap}", EdgeType.PROVIDES)
            if e and abs(float(e["level"]) - float(e["seed_level"])) >= 0.005:
                drift.append({
                    "model": m.id, "capability": cap,
                    "seed": round(float(e["seed_level"]), 4),
                    "current": round(float(e["level"]), 4),
                    "observations": int(e.get("observations", 0)),
                })
    return {
        "stats": r.kg.stats(),
        "ladder": r.kg.ladder(),
        "models": [
            {"id": m.id, "rung": m.rung, "context_window": m.context_window,
             "pricing": m.pricing, "config_surface": m.config_surface}
            for m in r.kg.models()
        ],
        "learned_drift": drift,
        "posteriors": [
            {"model": mid, "task": t, "mean": p.mean, "n": p.n}
            for mid, t, p in r.learning.posteriors()
        ],
        "mode": "dry-run" if r.dry_run else "live",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    r = get_router()
    return {"ok": True, "mode": "dry-run" if r.dry_run else "live",
            "models": len(r.kg.models())}
