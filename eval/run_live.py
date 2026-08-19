#!/usr/bin/env python3
"""Measure the LLM classifier -- the component that handles ~97% of real
classification and, until now, had never been executed.

Makes one Haiku call per query. Reports accuracy against labels, per-class
recall, and actual cost. This is a MEASUREMENT: per eval/README.md, changing
the classifier after reading a --test result retires that result.

    python eval/run_live.py --test [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_fabric import classifier, features as F  # noqa: E402
from decision_fabric.executor import Executor  # noqa: E402
from decision_fabric.seed import Fabric  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    path = HERE / f"{a.set}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if a.limit:
        rows = rows[: a.limit]

    fab = Fabric()
    ex = Executor(dry_run=False)
    spec = fab.kg.model(fab.meta_models["classifier"])
    task_types = sorted(fab.ontology["task_types"])
    domains = sorted(fab.ontology["domains"])
    out_classes = sorted(fab.output_classes)

    print(f"live LLM classification: {len(rows)} queries from {path.name} "
          f"using {spec.id}\n")

    total_cost = 0.0
    correct = 0
    errors = []
    by = collections.defaultdict(lambda: [0, 0])

    for i, r in enumerate(rows, 1):
        f = F.extract(r["q"], fab.signals, scope_cfg=fab.scope_signals)
        f, cost = classifier.refine(
            f, ex, spec, fab.modifiers,
            task_types=task_types, domains=domains, output_classes=out_classes,
        )
        total_cost += cost
        ok = f.task_type == r["label"]
        correct += ok
        by[r["label"]][1] += 1
        by[r["label"]][0] += ok
        if not ok:
            errors.append((r["q"], r["label"], f.task_type, f.task_confidence))
        if not a.quiet:
            mark = "ok " if ok else "MISS"
            print(f"  {i:>3}/{len(rows)} {mark} {r['label']:<20} -> {f.task_type:<20} "
                  f"${total_cost:.5f}")

    n = len(rows)
    print(f"\nLLM classifier accuracy: {correct}/{n} = {100*correct/n:.1f}%")
    print(f"total cost             : ${total_cost:.5f}  (${total_cost/n:.6f} per query)")

    print("\nPER-CLASS RECALL")
    for label in sorted(by):
        ok, tot = by[label]
        print(f"  {label:<22}{ok:>2}/{tot:<3} {100*ok/tot:>5.0f}%  {'#'*int(round(10*ok/tot))}")

    if errors:
        print(f"\nMISCLASSIFICATIONS ({len(errors)})")
        for q, e, g, c in errors:
            print(f"  {e} -> {g}  | {q[:74]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
