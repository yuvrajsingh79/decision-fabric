# Evaluation discipline

Three sets, three different jobs. Mixing them is how a classifier comes to
report 100% accuracy and deliver 32%.

| Set | File | Status | May be used for |
|---|---|---|---|
| **Train** | `../demo/queries.jsonl` | contaminated | Demos and cost illustration only. Patterns were written against it; its accuracy number is meaningless. |
| **Dev** | `dev.jsonl` | contaminated (60 q) | Iteration. Tune against this freely, and expect its numbers to be optimistic. |
| **Test** | `test.jsonl` | **SEALED** (90 q) | Reporting. Measure, report, do not tune. |

## The seal

`test.sha256` pins the content hash of `test.jsonl`. `eval/run.py --test` verifies
it before scoring and refuses to run if it drifts. This does not prevent tuning
against the set — nothing can — but it makes any edit to the set visible in
review, and it makes "we changed the test until it passed" an explicit act
rather than an accident.

**Rule: if a code change is made after reading a `--test` result, that result is
retired.** Re-seal with a new set of fresh queries; do not re-report the old
number. The point of the seal is to make that rule enforceable, not optional.

## Running

    python eval/run.py --dev      # iterate here
    python eval/run.py --test     # report from here, once
