#!/usr/bin/env python3
"""Classification + confidence-gate evaluation with train/dev/test discipline.

The headline metric is NOT raw accuracy. The regex layer is a spend gate, not a
classifier: its job is to know when it does not know, so the fabric can decide
whether to pay for an LLM classification. A gate is sound when

  * accuracy above the threshold is high    (committing is safe), and
  * it commits often enough to be worth having.

Reported alongside: expected calibration error, per-class recall, and the
silent-error count -- confident wrong answers, the only errors that reach
production unchecked.

    python eval/run.py --dev     # iterate here
    python eval/run.py --test    # report here, once (verifies the seal)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_fabric import features as F  # noqa: E402
from decision_fabric.seed import Fabric  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


def verify_seal(path: pathlib.Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    seal = HERE / "test.sha256"
    if not seal.is_file():
        return f"UNSEALED ({digest[:16]}…)"
    expected = seal.read_text().strip()
    if digest != expected:
        print(f"\n!! SEAL BROKEN for {path.name}", file=sys.stderr)
        print(f"   expected {expected[:16]}…  got {digest[:16]}…", file=sys.stderr)
        print("   The sealed test set was modified. Any number produced from it is "
              "not a held-out measurement.\n", file=sys.stderr)
        raise SystemExit(2)
    return f"verified {digest[:16]}…"


def evaluate(rows: list[dict], fab: Fabric) -> list[tuple[str, str, str, float]]:
    out = []
    for r in rows:
        f = F.extract(r["q"], fab.signals, scope_cfg=fab.scope_signals)
        out.append((r["q"], r["label"], f.task_type, f.task_confidence))
    return out


def expected_calibration_error(res, bins=5) -> float:
    """Mean gap between stated confidence and observed accuracy, weighted by bin size."""
    total, n = 0.0, len(res)
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        band = [r for r in res if lo <= r[3] < hi or (i == bins - 1 and r[3] == 1.0)]
        if not band:
            continue
        acc = sum(1 for _, e, g, _ in band if e == g) / len(band)
        conf = sum(r[3] for r in band) / len(band)
        total += len(band) / n * abs(acc - conf)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true")
    g.add_argument("--test", action="store_true")
    g.add_argument("--set", help="arbitrary jsonl")
    ap.add_argument("--errors", action="store_true", help="list every misclassification")
    a = ap.parse_args()

    if a.test:
        path, kind = HERE / "test.jsonl", "TEST (sealed)"
        seal = verify_seal(path)
    elif a.dev:
        path, kind, seal = HERE / "dev.jsonl", "DEV (contaminated)", "n/a"
    else:
        path, kind, seal = pathlib.Path(a.set), "custom", "n/a"

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    res = evaluate(rows, Fabric())
    n = len(res)
    correct = sum(1 for _, e, g, _ in res if e == g)

    print(f"set   : {kind}  {path.name}  ({n} queries, "
          f"{len(set(r[1] for r in res))} classes)")
    print(f"seal  : {seal}")
    if a.dev:
        print("        NOTE: patterns were tuned against this set. Numbers are optimistic.")
    print()
    print(f"raw accuracy            {correct}/{n} = {100*correct/n:.1f}%")
    print(f"expected calib. error   {expected_calibration_error(res):.3f}   "
          f"(0 = stated confidence matches observed accuracy)")
    print()

    fab = Fabric()
    hi = [r for r in res if fab.may_commit(r[3])[0]]
    lo = [r for r in res if not fab.may_commit(r[3])[0]]
    hi_ok = sum(1 for _, e, g, _ in hi if e == g)
    lo_ok = sum(1 for _, e, g, _ in lo if e == g)
    silent = [r for r in hi if r[1] != r[2]]

    floor = fab.classification_cfg.get("min_commit_precision", "n/a")
    print(f"CALIBRATED GATE (commit only where measured precision >= {floor})")
    print(f"  commit (no LLM call)   {len(hi):>3}/{n} = {100*len(hi)/n:>4.0f}%   "
          f"accuracy {100*hi_ok/len(hi) if hi else 0:>5.1f}%")
    print(f"  escalate to LLM        {len(lo):>3}/{n} = {100*len(lo)/n:>4.0f}%   "
          f"accuracy {100*lo_ok/len(lo) if lo else 0:>5.1f}%")
    print(f"  silent errors          {len(silent):>3}      "
          f"(confidently wrong, nothing downstream checks these)")
    if hi and lo:
        print(f"  separation             {100*hi_ok/len(hi) - 100*lo_ok/len(lo):+.1f} points")

    print("\nPER-CLASS RECALL (heuristic alone)")
    by = collections.defaultdict(lambda: [0, 0])
    for _, e, gg, _ in res:
        by[e][1] += 1
        by[e][0] += (e == gg)
    for label in sorted(by):
        ok, tot = by[label]
        bar = "#" * int(round(10 * ok / tot))
        print(f"  {label:<22}{ok:>2}/{tot:<3} {100*ok/tot:>5.0f}%  {bar}")

    if silent:
        print(f"\nSILENT ERRORS ({len(silent)}) — the ones that matter")
        for q, e, gg, c in silent:
            print(f"  conf={c:.2f}  {e} -> {gg}\n      {q[:84]}")

    if a.errors:
        print("\nALL MISCLASSIFICATIONS")
        for q, e, gg, c in res:
            if e != gg:
                print(f"  conf={c:.2f}  {e} -> {gg}  | {q[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
