#!/usr/bin/env python3
"""Measure heuristic classification accuracy and, more importantly, whether the
confidence score is CALIBRATED.

Confidence is a spend gate: at >= 0.55 the fabric commits to the heuristic guess
and skips the paid LLM classifier. That gate is only sound if high confidence
actually means high accuracy. If it does not, the fabric is confidently
misrouting and calling the resulting underspend a saving.

    python demo/eval.py [--set demo/eval_classification.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from decision_fabric import features as F  # noqa: E402
from decision_fabric.classifier import REFINE_BELOW_CONFIDENCE  # noqa: E402
from decision_fabric.seed import Fabric  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=str(pathlib.Path(__file__).parent / "eval_classification.jsonl"))
    a = ap.parse_args()

    fab = Fabric()
    rows = [json.loads(l) for l in pathlib.Path(a.set).read_text().splitlines() if l.strip()]

    results = []
    for r in rows:
        f = F.extract(r["q"], fab.signals, scope_cfg=fab.scope_signals)
        results.append((r["q"], r["label"], f.task_type, f.task_confidence))

    n = len(results)
    correct = sum(1 for _, exp, got, _ in results if exp == got)
    print(f"held-out set: {n} queries, {len(set(r[1] for r in results))} task types\n")
    print(f"OVERALL heuristic accuracy: {correct}/{n} = {100*correct/n:.1f}%\n")

    # --- the number that actually matters: the spend gate ---
    hi = [r for r in results if r[3] >= REFINE_BELOW_CONFIDENCE]
    lo = [r for r in results if r[3] < REFINE_BELOW_CONFIDENCE]
    hi_ok = sum(1 for _, e, g, _ in hi if e == g)
    lo_ok = sum(1 for _, e, g, _ in lo if e == g)
    print(f"CONFIDENCE GATE (threshold {REFINE_BELOW_CONFIDENCE})")
    print(f"  >= {REFINE_BELOW_CONFIDENCE}  commit to heuristic, no LLM call : "
          f"{len(hi):>3} queries ({100*len(hi)/n:.0f}%)  accuracy {100*hi_ok/len(hi) if hi else 0:.1f}%")
    print(f"  <  {REFINE_BELOW_CONFIDENCE}  pay for the LLM classifier      : "
          f"{len(lo):>3} queries ({100*len(lo)/n:.0f}%)  accuracy {100*lo_ok/len(lo) if lo else 0:.1f}%")
    if hi and lo:
        gap = 100*hi_ok/len(hi) - 100*lo_ok/len(lo)
        verdict = "informative" if gap > 10 else "NOT informative — the gate is not separating anything"
        print(f"  separation: {gap:+.1f} points -> confidence is {verdict}")

    # --- calibration: does stated confidence match observed accuracy? ---
    print("\nCALIBRATION")
    print(f"  {'confidence band':<18}{'n':>4}{'stated':>9}{'actual':>9}{'error':>9}")
    bands = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo_b, hi_b in bands:
        band = [r for r in results if lo_b <= r[3] < hi_b]
        if not band:
            continue
        ok = sum(1 for _, e, g, _ in band if e == g)
        stated = sum(r[3] for r in band) / len(band)
        actual = ok / len(band)
        print(f"  [{lo_b:.2f}, {hi_b:.2f})      {len(band):>4}{stated:>9.2f}{actual:>9.2f}"
              f"{actual - stated:>+9.2f}")

    # --- where it goes wrong ---
    errs = [(q, e, g, c) for q, e, g, c in results if e != g]
    if errs:
        print(f"\nMISCLASSIFICATIONS ({len(errs)})")
        pairs = collections.Counter((e, g) for _, e, g, _ in errs)
        for (exp, got), cnt in pairs.most_common():
            print(f"  {exp} -> {got}  ({cnt}x)")
        print()
        for q, e, g, c in errs:
            flag = "!! HIGH CONF" if c >= REFINE_BELOW_CONFIDENCE else "   (low conf, LLM would fire)"
            print(f"  {flag}  conf={c:.2f}  expected {e}, got {g}")
            print(f"      {q[:88]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
