"""CLI: `python -m decision_fabric.cli <command>`"""
from __future__ import annotations

import argparse
import json
import sys

from .env import load_dotenv
from .explain import render, render_savings
from .router import Router


def _router(a: argparse.Namespace) -> Router:
    load_dotenv()
    dry = None
    if getattr(a, "dry_run", False):
        dry = True
    elif getattr(a, "live", False):
        dry = False
    return Router(
        config_dir=getattr(a, "config", None),
        db_path=a.db,
        dry_run=dry,
        use_llm_classifier=not getattr(a, "no_classifier", False),
    )


def cmd_route(a: argparse.Namespace) -> int:
    r = _router(a)
    d = r.route(
        a.query,
        execute=not a.plan_only,
        policy=a.policy,
        latency_slo=a.slo,
        domain=a.domain,
        task_type=a.task,
        context_tokens=a.context_tokens,
        stable_context_tokens=a.stable_tokens,
        zero_data_retention=a.zdr,
        learn=not a.no_learn,
    )
    print(json.dumps(d.to_dict(), indent=2) if a.json
          else render(d, show_trace=a.trace, show_answer=a.show_answer))
    if not a.json:
        print(f"\nmode: {'DRY-RUN (no API calls)' if r.dry_run else 'LIVE'}")
    r.close()
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    r = _router(a)
    rep = r.telemetry.savings_report()
    print(json.dumps(rep, indent=2) if a.json else render_savings(rep))
    r.close()
    return 0


def cmd_graph(a: argparse.Namespace) -> int:
    r = _router(a)
    if a.dot:
        print(r.kg.to_dot())
    else:
        print(json.dumps(r.kg.stats(), indent=2))
        print("\nescalation ladder:", " -> ".join(r.kg.ladder()))
        print("\nlearned posteriors:")
        rows = r.learning.posteriors()
        if not rows:
            print("  (none yet — run `demo/run_demo.py` or route some queries)")
        for model, task, p in rows:
            print(f"  {model:<22}{task:<24} mean={p.mean:.3f}  n={p.n}  s={p.successes}")
    r.close()
    return 0


def cmd_replay(a: argparse.Namespace) -> int:
    """Re-route a stored query against the *current* graph to see what changed."""
    r = _router(a)
    row = r.telemetry.conn.execute(
        "SELECT * FROM routes WHERE id=?", (a.route_id,)
    ).fetchone()
    if row is None:
        print(f"no route with id {a.route_id}", file=sys.stderr)
        r.close()
        return 1
    print(f"stored: {row['first_model']} -> {row['final_model']}  ({row['plan']})")
    print("note: only a 160-char query preview is stored, and attached-context sizes are not —"
          "\n      caching decisions in particular will not reproduce exactly.")
    d = r.route(row["query_preview"], execute=False, policy=row["policy"],
                latency_slo=row["slo"], learn=False)
    print(f"now:    {d.selection.first_model}  ({d.selection.first_plan.summary()})")
    r.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("decision-fabric")
    p.add_argument("--db", default="./decision_fabric.db")
    p.add_argument("--config", default=None, help="path to config/ dir")
    p.add_argument("--dry-run", action="store_true", help="force simulation, never call the API")
    p.add_argument("--live", action="store_true", help="force live API calls")
    p.add_argument("--json", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("route", help="route one query")
    r.add_argument("query")
    r.add_argument("--policy", default=None, choices=["economy", "balanced", "quality", "critical"])
    r.add_argument("--slo", default="interactive",
                   choices=["realtime", "interactive", "background", "batch"])
    r.add_argument("--domain", default=None)
    r.add_argument("--task", default=None, help="override the task type")
    r.add_argument("--context-tokens", type=int, default=0)
    r.add_argument("--stable-tokens", type=int, default=0,
                   help="reusable prefix size, for cache planning")
    r.add_argument("--zdr", action="store_true",
                   help="org is on zero data retention — exclude models that require 30-day retention")
    r.add_argument("--plan-only", action="store_true", help="decide but do not execute")
    r.add_argument("--trace", action="store_true", help="print the full decision trace")
    r.add_argument("--show-answer", action="store_true")
    r.add_argument("--no-learn", action="store_true")
    r.add_argument("--no-classifier", action="store_true")
    r.set_defaults(func=cmd_route)

    rep = sub.add_parser("report", help="cumulative savings vs the no-router baseline")
    rep.set_defaults(func=cmd_report)

    g = sub.add_parser("graph", help="inspect the knowledge graph")
    g.add_argument("--dot", action="store_true", help="emit Graphviz DOT")
    g.set_defaults(func=cmd_graph)

    rp = sub.add_parser("replay", help="re-decide a stored route against the current graph")
    rp.add_argument("route_id", type=int)
    rp.set_defaults(func=cmd_replay)

    a = p.parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    raise SystemExit(main())
