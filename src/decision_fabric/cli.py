"""CLI: `python -m decision_fabric.cli <command>`"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

from .env import load_dotenv
from .explain import render, render_savings
from .router import Router


def _router(a: argparse.Namespace) -> Router:
    load_dotenv()
    # Spending is opt-in. Without --live nothing reaches the API, even when
    # credentials are present.
    dry = False if getattr(a, "live", False) else True
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


def cmd_costs(a: argparse.Namespace) -> int:
    """Where money goes: every component that can make an API call."""
    r = _router(a)
    fab, kg = r.fabric, r.kg
    from .pricing import project_cost
    hai = kg.model(fab.meta_models["classifier"])
    mods = fab.modifiers

    cls_cost = project_cost(hai, mods, fresh_input_tokens=320, output_tokens=80,
                            effort="none").total_usd
    ver_cost = project_cost(hai, mods, fresh_input_tokens=700, output_tokens=120,
                            effort="none").total_usd

    print("COMPONENTS THAT MAKE API CALLS")
    print(f"{'#':<3}{'component':<22}{'model':<18}{'when':<38}{'typical $':>10}")
    print("-" * 92)
    print(f"{'1':<3}{'classifier.refine':<22}{hai.id:<18}"
          f"{'gate says escalate (~97% of queries)':<38}{cls_cost:>10.6f}")
    print(f"{'2':<3}{'executor.execute':<22}{'the ROUTED model':<18}"
          f"{'every route unless --plan-only':<38}{'varies':>10}")
    print(f"{'3':<3}{'verifier._llm':<22}{hai.id:<18}"
          f"{'policy quality/critical only':<38}{ver_cost:>10.6f}")
    print(f"{'-':<3}{'escalation':<22}{'next rung up':<18}"
          f"{'verification failed':<38}{'varies':>10}")
    print("\n#2 is the answer to the query itself, and it dominates. #1 and #3")
    print("are the router's own overhead.")

    print("COST OF ONE ANSWER, BY MODEL AND EFFORT (medium task, ~1800 output tokens)")
    oc = fab.output_classes["medium"]
    print(f"  {'model':<20}" + "".join(f"{e:>10}" for e in ["none", "low", "medium", "high", "xhigh", "max"]))
    for spec in kg.models():
        row = f"  {spec.id:<20}"
        for eff in ["none", "low", "medium", "high", "xhigh", "max"]:
            cst = project_cost(spec, mods, fresh_input_tokens=1000,
                               output_tokens=oc["expected_output_tokens"], effort=eff).total_usd
            row += f"{cst:>10.4f}"
        print(row)
    print("\nBatch API halves every figure. Cached input reads at 10%.")

    spent = r.telemetry.conn.execute(
        "SELECT COUNT(*) n, SUM(actual_cost_usd) usd FROM routes WHERE mode='live'"
    ).fetchone()
    print(f"\nRECORDED LIVE SPEND IN {a.db}: "
          f"{spent['n'] or 0} routes, ${spent['usd'] or 0.0:.5f}")
    print("(dry-run routes are excluded — they cost nothing)")
    r.close()
    return 0


def cmd_spend(a: argparse.Namespace) -> int:
    """Total live spend across every entry point, from the unified ledger."""
    from .spend import DEFAULT_LEDGER, SpendLedger
    path = pathlib.Path(a.ledger) if a.ledger else DEFAULT_LEDGER
    if not path.is_file():
        print(f"no ledger at {path} — nothing has been spent through this tool yet.")
        return 0
    led = SpendLedger(path)
    t = led.totals()
    print(f"LEDGER {path}\n")
    print(f"  total spend      ${t.usd:.5f}")
    print(f"  API calls        {t.calls}")
    print(f"  tokens           {t.input_tokens:,} in / {t.output_tokens:,} out")
    if a.budget:
        pct = 100 * t.usd / a.budget
        print(f"  of ${a.budget:.2f} budget  {pct:.2f}%  (${a.budget - t.usd:.5f} left)")

    print("\n  BY COMPONENT")
    for r in led.by("component"):
        print(f"    {r['key']:<16}{r['calls']:>5} calls  ${r['usd']:>10.5f}")
    print("\n  BY MODEL")
    for r in led.by("model_id"):
        print(f"    {r['key']:<22}{r['calls']:>5} calls  ${r['usd']:>10.5f}")

    if a.recent:
        print("\n  MOST RECENT")
        for r in led.recent(a.recent):
            ts = datetime.datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S")
            print(f"    {ts}  {r['component']:<14}{r['model_id']:<20}"
                  f"${r['usd']:>9.5f}  {(r['note'] or '')[:36]}")

    print("\n  This ledger records only calls made through this tool. The")
    print("  Anthropic Console is authoritative for your actual balance:")
    print("  https://platform.claude.com/settings/usage")
    led.close()
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


def _add_global_flags(p: argparse.ArgumentParser, *, suppress: bool = False) -> None:
    """Global flags. On subparsers, defaults are SUPPRESSed so that an unset
    flag after the subcommand does not silently overwrite the value parsed
    before it -- which would make `--db X costs` and `costs --db X` disagree."""
    d = (lambda v: argparse.SUPPRESS) if suppress else (lambda v: v)
    p.add_argument("--db", default=d("./decision_fabric.db"))
    p.add_argument("--config", default=d(None), help="path to config/ dir")
    p.add_argument("--dry-run", action="store_true", default=d(False),
                   help="simulate, never call the API (this is the DEFAULT)")
    p.add_argument("--live", action="store_true", default=d(False),
                   help="REQUIRED to make real API calls and spend money")
    p.add_argument("--json", action="store_true", default=d(False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("decision-fabric")
    _add_global_flags(p)
    # Repeated on every subparser so `--db` works either side of the
    # subcommand. Argparse otherwise silently rejects the natural ordering.
    common = argparse.ArgumentParser(add_help=False)
    _add_global_flags(common, suppress=True)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("route", parents=[common], help="route one query")
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

    sp = sub.add_parser("spend", parents=[common],
                        help="TOTAL live spend across every entry point")
    sp.add_argument("--ledger", default=None, help="path to the spend ledger")
    sp.add_argument("--budget", type=float, default=None, help="show % of this USD budget")
    sp.add_argument("--recent", type=int, default=0, help="list the N most recent calls")
    sp.set_defaults(func=cmd_spend)

    cst = sub.add_parser("costs", parents=[common], help="where money goes: every component that can bill")
    cst.set_defaults(func=cmd_costs)

    rep = sub.add_parser("report", parents=[common], help="cumulative savings vs the no-router baseline")
    rep.set_defaults(func=cmd_report)

    g = sub.add_parser("graph", parents=[common], help="inspect the knowledge graph")
    g.add_argument("--dot", action="store_true", help="emit Graphviz DOT")
    g.set_defaults(func=cmd_graph)

    rp = sub.add_parser("replay", parents=[common], help="re-decide a stored route against the current graph")
    rp.add_argument("route_id", type=int)
    rp.set_defaults(func=cmd_replay)

    a = p.parse_args(argv)
    return int(a.func(a))


if __name__ == "__main__":
    raise SystemExit(main())
