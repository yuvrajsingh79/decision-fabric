"""A single append-only ledger of every live API call this tool makes.

Route telemetry lives per-database and only covers calls made through
`Router.route()`. That misses the conformance check, the classifier evaluation,
and any second database — so "what have I spent" had no answer. Every live call
now lands here regardless of entry point.

This is the tool's own record. It is NOT authoritative: only the Anthropic
Console knows your real balance, and anything you run outside this tool will
never appear here.
"""
from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_LEDGER = Path(
    os.environ.get("DECISION_FABRIC_LEDGER", "./.decision_fabric_spend.db")
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    component     TEXT NOT NULL,   -- routed_model | classifier | verifier | conformance
    model_id      TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read    INTEGER NOT NULL DEFAULT 0,
    cache_write   INTEGER NOT NULL DEFAULT 0,
    usd           REAL NOT NULL,
    note          TEXT
);
CREATE INDEX IF NOT EXISTS idx_spend_ts ON spend(ts);
"""


@dataclass
class SpendTotals:
    calls: int
    usd: float
    input_tokens: int
    output_tokens: int


class SpendLedger:
    def __init__(self, path: Path | str = DEFAULT_LEDGER) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def record(
        self, component: str, model_id: str, usage: Any, usd: float, note: str = ""
    ) -> None:
        get = (lambda k: usage.get(k, 0)) if isinstance(usage, dict) else (
            lambda k: getattr(usage, k, 0) or 0
        )
        self.conn.execute(
            "INSERT INTO spend (ts, component, model_id, input_tokens, output_tokens,"
            " cache_read, cache_write, usd, note) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                time.time(), component, model_id,
                int(get("input_tokens")), int(get("output_tokens")),
                int(get("cache_read_input_tokens")), int(get("cache_creation_input_tokens")),
                float(usd), note[:200],
            ),
        )
        self.conn.commit()

    def totals(self) -> SpendTotals:
        r = self.conn.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(usd),0) usd,"
            " COALESCE(SUM(input_tokens),0) i, COALESCE(SUM(output_tokens),0) o FROM spend"
        ).fetchone()
        return SpendTotals(int(r["n"]), float(r["usd"]), int(r["i"]), int(r["o"]))

    def by(self, column: str) -> list[dict[str, Any]]:
        if column not in ("component", "model_id"):
            raise ValueError(column)
        rows = self.conn.execute(
            f"SELECT {column} k, COUNT(*) n, SUM(usd) usd, SUM(input_tokens) i,"
            f" SUM(output_tokens) o FROM spend GROUP BY {column} ORDER BY SUM(usd) DESC"
        ).fetchall()
        return [
            {"key": r["k"], "calls": r["n"], "usd": float(r["usd"]),
             "input_tokens": int(r["i"]), "output_tokens": int(r["o"])}
            for r in rows
        ]

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT ts, component, model_id, output_tokens, usd, note"
            " FROM spend ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self.conn.close()
