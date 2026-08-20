"""Durable record of every routing decision and what it actually cost.

Two things live here: the audit trail (why did we pick that, what did it cost)
and the observation stream the learning loop feeds on.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DB = Path("./decision_fabric.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS routes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL NOT NULL,
    query_hash     TEXT NOT NULL,
    query_preview  TEXT NOT NULL,
    task_type      TEXT NOT NULL,
    domain         TEXT NOT NULL,
    signals        TEXT NOT NULL,
    policy         TEXT NOT NULL,
    slo            TEXT NOT NULL,
    first_model    TEXT NOT NULL,
    final_model    TEXT NOT NULL,
    plan           TEXT NOT NULL,
    cascade        INTEGER NOT NULL,
    escalated      INTEGER NOT NULL,
    accepted       INTEGER NOT NULL,
    est_cost_usd   REAL NOT NULL,
    actual_cost_usd REAL NOT NULL,
    baseline_cost_usd REAL NOT NULL,
    input_tokens   INTEGER NOT NULL,
    output_tokens  INTEGER NOT NULL,
    mode           TEXT NOT NULL,
    trace          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_routes_task ON routes(task_type);
CREATE INDEX IF NOT EXISTS idx_routes_ts ON routes(ts);

CREATE TABLE IF NOT EXISTS observations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    model_id   TEXT NOT NULL,
    task_type  TEXT NOT NULL,
    success    INTEGER NOT NULL,
    source     TEXT NOT NULL,       -- verifier | escalation | human
    route_id   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_obs_pair ON observations(model_id, task_type);

-- Evidence about TASK DEFINITIONS rather than about models: for the capability
-- that was the binding constraint on the selected model, how much headroom did
-- it have, and did the answer pass?
CREATE TABLE IF NOT EXISTS requirement_feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    task_type  TEXT NOT NULL,
    capability TEXT NOT NULL,
    model_id   TEXT NOT NULL,
    margin     REAL NOT NULL,
    required   REAL NOT NULL,
    success    INTEGER NOT NULL,
    route_id   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_reqfb ON requirement_feedback(task_type, capability);
"""


@dataclass
class RouteRecord:
    query: str
    task_type: str
    domain: str
    signals: list[str]
    policy: str
    slo: str
    first_model: str
    final_model: str
    plan: str
    cascade: bool
    escalated: bool
    accepted: bool
    est_cost_usd: float
    actual_cost_usd: float
    baseline_cost_usd: float
    input_tokens: int
    output_tokens: int
    mode: str
    trace: list[str]


class Telemetry:
    def __init__(self, db_path: Path | str = DEFAULT_DB) -> None:
        self.path = Path(db_path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------- writes ----------

    def record_route(self, r: RouteRecord) -> int:
        cur = self.conn.execute(
            """INSERT INTO routes (ts, query_hash, query_preview, task_type, domain, signals,
                                   policy, slo, first_model, final_model, plan, cascade, escalated,
                                   accepted, est_cost_usd, actual_cost_usd, baseline_cost_usd,
                                   input_tokens, output_tokens, mode, trace)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(),
                hashlib.sha256(r.query.encode()).hexdigest()[:16],
                r.query[:160],
                r.task_type, r.domain, json.dumps(r.signals), r.policy, r.slo,
                r.first_model, r.final_model, r.plan,
                int(r.cascade), int(r.escalated), int(r.accepted),
                r.est_cost_usd, r.actual_cost_usd, r.baseline_cost_usd,
                r.input_tokens, r.output_tokens, r.mode, json.dumps(r.trace),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def record_observation(
        self, model_id: str, task_type: str, success: bool, source: str, route_id: int | None = None
    ) -> None:
        self.conn.execute(
            "INSERT INTO observations (ts, model_id, task_type, success, source, route_id)"
            " VALUES (?,?,?,?,?,?)",
            (time.time(), model_id, task_type, int(success), source, route_id),
        )
        self.conn.commit()

    # ---------- reads ----------

    def record_requirement_feedback(self, task_type: str, capability: str, model_id: str,
                                    margin: float, required: float, success: bool,
                                    route_id: int | None = None) -> None:
        self.conn.execute(
            "INSERT INTO requirement_feedback (ts, task_type, capability, model_id,"
            " margin, required, success, route_id) VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), task_type, capability, model_id, float(margin),
             float(required), int(success), route_id),
        )
        self.conn.commit()

    def requirement_evidence(self) -> list[dict[str, Any]]:
        """Per (task, capability): how often a thin margin failed, and how often
        a generous margin succeeded."""
        rows = self.conn.execute(
            "SELECT task_type, capability, COUNT(*) n, SUM(success) s,"
            " AVG(margin) avg_margin, MIN(margin) min_margin"
            " FROM requirement_feedback GROUP BY task_type, capability"
        ).fetchall()
        return [dict(r) for r in rows]

    def requirement_rows(self, task_type: str, capability: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT margin, success FROM requirement_feedback"
            " WHERE task_type=? AND capability=?", (task_type, capability)
        ).fetchall()
        return [dict(r) for r in rows]

    def counts(self, model_id: str, task_type: str) -> tuple[int, int]:
        row = self.conn.execute(
            "SELECT SUM(success) AS s, COUNT(*) AS n FROM observations"
            " WHERE model_id=? AND task_type=?",
            (model_id, task_type),
        ).fetchone()
        return int(row["s"] or 0), int(row["n"] or 0)

    def all_pairs(self) -> list[tuple[str, str, int, int]]:
        rows = self.conn.execute(
            "SELECT model_id, task_type, SUM(success) AS s, COUNT(*) AS n"
            " FROM observations GROUP BY model_id, task_type"
        ).fetchall()
        return [(r["model_id"], r["task_type"], int(r["s"] or 0), int(r["n"])) for r in rows]

    def savings_report(self) -> dict[str, Any]:
        row = self.conn.execute(
            """SELECT COUNT(*) n,
                      SUM(actual_cost_usd) actual,
                      SUM(baseline_cost_usd) baseline,
                      SUM(escalated) escalations,
                      SUM(cascade) cascades,
                      SUM(input_tokens) in_tok,
                      SUM(output_tokens) out_tok
               FROM routes"""
        ).fetchone()
        n = int(row["n"] or 0)
        actual = float(row["actual"] or 0.0)
        baseline = float(row["baseline"] or 0.0)
        by_model = self.conn.execute(
            "SELECT final_model, COUNT(*) n, SUM(actual_cost_usd) usd"
            " FROM routes GROUP BY final_model ORDER BY n DESC"
        ).fetchall()
        by_task = self.conn.execute(
            """SELECT task_type, COUNT(*) n,
                      SUM(actual_cost_usd) actual, SUM(baseline_cost_usd) baseline
               FROM routes GROUP BY task_type ORDER BY (SUM(baseline_cost_usd)-SUM(actual_cost_usd)) DESC"""
        ).fetchall()
        return {
            "routes": n,
            "actual_usd": round(actual, 6),
            "baseline_usd": round(baseline, 6),
            "saved_usd": round(baseline - actual, 6),
            "saved_pct": round(100 * (baseline - actual) / baseline, 2) if baseline else 0.0,
            "cascades": int(row["cascades"] or 0),
            "escalations": int(row["escalations"] or 0),
            "escalation_rate": round((row["escalations"] or 0) / n, 3) if n else 0.0,
            "input_tokens": int(row["in_tok"] or 0),
            "output_tokens": int(row["out_tok"] or 0),
            "by_model": [
                {"model": r["final_model"], "routes": r["n"], "usd": round(r["usd"], 6)}
                for r in by_model
            ],
            "by_task": [
                {
                    "task": r["task_type"], "routes": r["n"],
                    "actual_usd": round(r["actual"], 6),
                    "baseline_usd": round(r["baseline"], 6),
                    "saved_usd": round(r["baseline"] - r["actual"], 6),
                }
                for r in by_task
            ],
        }
