#!/usr/bin/env python3
"""Route a representative enterprise query mix through the fabric and report
what it cost versus sending everything to the flagship.

    python demo/run_demo.py                # dry-run, no API calls, deterministic
    python demo/run_demo.py --live         # real Claude calls (costs money)
    python demo/run_demo.py --rounds 3     # show the graph adapting over rounds
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decision_fabric.env import load_dotenv  # noqa: E402
from decision_fabric.explain import render, render_savings  # noqa: E402
from decision_fabric.router import Router  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def load(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="make real API calls")
    ap.add_argument("--rounds", type=int, default=1, help="replay the corpus N times to show learning")
    ap.add_argument("--db", default=str(HERE / "demo.db"))
    ap.add_argument("--queries", default=str(HERE / "queries.jsonl"))
    ap.add_argument("--detail", action="store_true", help="print the full report per query")
    ap.add_argument("--keep-db", action="store_true", help="do not wipe prior telemetry")
    ap.add_argument("--tier", default=None, choices=["cheap", "mid", "heavy"],
                    help="only run queries at or below this cost tier (live corpus only)")
    ap.add_argument("--max-spend", type=float, default=None,
                    help="abort before a query that would push actual spend past this USD amount")
    a = ap.parse_args()

    load_dotenv()
    db = pathlib.Path(a.db)
    if db.exists() and not a.keep_db:
        db.unlink()

    rows = load(pathlib.Path(a.queries))
    if a.tier:
        allowed = {"cheap": {"cheap"}, "mid": {"cheap", "mid"},
                   "heavy": {"cheap", "mid", "heavy"}}[a.tier]
        rows = [r for r in rows if r.get("tier", "cheap") in allowed]
    r = Router(db_path=db, dry_run=None if a.live else True)
    print(f"mode: {'LIVE — real API calls' if not r.dry_run else 'DRY-RUN — simulated, no API calls'}")
    print(f"corpus: {len(rows)} queries x {a.rounds} round(s)\n")

    spent = [0.0]
    header = f"{'#':>3}  {'task':<22}{'model':<20}{'plan':<46}{'routed':>11}{'baseline':>11}{'save':>8}"
    for rnd in range(1, a.rounds + 1):
        if a.rounds > 1:
            print(f"\n=== round {rnd} ===")
        print(header)
        print("-" * len(header))
        for i, row in enumerate(rows, 1):
            if a.max_spend is not None and spent[0] >= a.max_spend:
                print(f"--- stopping: spend guard ${a.max_spend:.2f} reached "
                      f"(${spent[0]:.4f} spent, {len(rows) - i + 1} queries skipped) ---")
                break
            d = r.route(
                row["q"],
                policy=row.get("policy"),
                latency_slo=row.get("slo", "interactive"),
                context_tokens=row.get("context_tokens", 0),
                stable_context_tokens=row.get("stable_context_tokens", 0),
            )
            spent[0] += d.total_usd
            if a.detail:
                print(render(d, show_trace=False))
                continue
            flag = "^" if d.escalated else (">" if d.selection.cascade else " ")
            print(
                f"{i:>3}{flag} {d.features.task_type:<22}{d.chosen_model:<20}"
                f"{d.selection.first_plan.summary()[:44]:<46}"
                f"{d.total_usd:>11.5f}{d.baseline_usd:>11.5f}{d.saved_pct:>7.0f}%"
            )

    print()
    print(render_savings(r.telemetry.savings_report()))
    print("\nlegend:  > cascade (cheap probe first)   ^ escalated after verification failed")
    if r.dry_run:
        print("note: dry-run token counts are simulated; set ANTHROPIC_API_KEY and pass --live for real ones.")
    r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
