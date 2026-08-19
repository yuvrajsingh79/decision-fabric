# Command reference

Setup, once:

```bash
source .venv/bin/activate
pip install -e .          # makes `decision-fabric` and `import decision_fabric` work
```

After this you never need `PYTHONPATH=src`.

---

## Cost safety

**Nothing spends money unless you pass `--live`.** That is true even though
credentials are present on this machine. `pytest` cannot spend at all — a
tripwire in `tests/conftest.py` blocks live client construction and fails the
test if anything tries.

| | |
|---|---|
| `--live` absent | simulated, $0 |
| `--live` present | real API calls, real money |
| `--max-spend N` | abort before exceeding N USD (demo runner) |

Check total spend at any time:

```bash
decision-fabric spend --budget 5          # everything, across all entry points
decision-fabric spend --recent 10         # last 10 billed calls
```

`spend` reads a unified ledger (`.decision_fabric_spend.db`) that every live
call writes to — routed models, classifier, verifier, and the conformance
probes alike. `costs` is different: it shows the *price list* and one
database's route history.

The ledger sees only calls made through this tool. The
[Anthropic Console](https://platform.claude.com/settings/usage) is
authoritative for your real balance.

---

## FREE — no API calls, run these freely

### Route one query and see the whole decision
```bash
decision-fabric route "why is my checkout service throwing 502s?"
decision-fabric route "write a python project for a snake game" --trace
decision-fabric route "summarise this contract" --policy critical --slo background
decision-fabric route "classify these tickets" --policy economy --slo batch
decision-fabric route "review my auth code" --task code_review        # skip classification
decision-fabric route "answer from the attached doc" --context-tokens 30000 --stable-tokens 28000
decision-fabric route "prove this terminates" --zdr                   # zero-data-retention org
decision-fabric route "anything" --plan-only                          # decide, never execute
decision-fabric route "anything" --json                               # machine-readable
```

Useful flags: `--policy economy|balanced|quality|critical`,
`--slo realtime|interactive|background|batch`, `--domain legal|finance|healthcare|…`,
`--trace` (full decision trace), `--show-answer`, `--no-learn`, `--no-classifier`.

### Inspect the fabric
```bash
decision-fabric graph                          # node/edge counts, ladder, learned posteriors
decision-fabric graph --dot > fabric.dot       # Graphviz export
dot -Tsvg fabric.dot > fabric.svg              # needs graphviz installed
decision-fabric costs                          # who spends, and how much per model/effort
decision-fabric report --db demo/demo.db       # cumulative savings vs baseline
decision-fabric replay 3 --db demo/demo.db     # re-decide a stored route against today's graph
```

### Simulated corpus runs
```bash
python demo/run_demo.py                                    # 30-query mix + savings report
python demo/run_demo.py --rounds 5                         # watch the graph learn
python demo/run_demo.py --detail                           # full report per query
python demo/run_demo.py --queries demo/queries_live.jsonl --tier cheap
```

### Tests and evaluation
```bash
pytest tests/ -q                        # 40 tests, guaranteed $0
python eval/run.py --dev                # accuracy on the contaminated dev set
python eval/run.py --test               # accuracy on the SEALED test set
python eval/run.py --test --errors      # list every misclassification
python eval/calibrate.py                # regenerate config/calibration.yaml from dev data
```

---

## COSTS MONEY — each of these bills your account

Measured costs from this project's own live runs:

```bash
# Config-surface conformance: 20 calls, 4 models x 5 effort rungs.   ~$0.008
python demo/live_check.py

# LLM classifier accuracy on the 75-query sealed set.                ~$0.060
python eval/run_live.py --set test --quiet
python eval/run_live.py --set test --limit 5      # smoke test first  ~$0.004

# Seven cheap real queries end to end.                               ~$0.020
python demo/run_demo.py --queries demo/queries_live.jsonl --tier cheap --live --max-spend 0.25

# Adds 3 code-gen / creative queries on Sonnet 5.                    ~$0.050
python demo/run_demo.py --queries demo/queries_live.jsonl --tier mid --live --max-spend 0.25

# One real routed answer.                            varies — SEE `costs` FIRST
decision-fabric route "your question" --live --show-answer
```

### Expensive — think before running

```bash
# Routes to Opus 5 @ xhigh and Fable 5 @ max. Minutes per query.       ~$3.00
python demo/run_demo.py --queries demo/queries_live.jsonl --tier heavy --live --max-spend 3.50
```

A single Fable 5 answer at `max` effort is ~$0.64. Effort is a 6.4x multiplier
on output cost, and thinking tokens bill as output. `decision-fabric costs`
prints the full model x effort grid.

---

## HTTP service

```bash
pip install -r requirements-api.txt
uvicorn decision_fabric.api:app --reload                    # simulated (safe)
DECISION_FABRIC_LIVE=1 uvicorn decision_fabric.api:app      # real calls
```

```bash
curl -s localhost:8000/plan -H 'content-type: application/json' \
  -d '{"query":"review this migration","policy":"quality"}' | jq

curl -s localhost:8000/savings | jq
curl -s localhost:8000/graph | jq '.learned_drift'
```

`POST /plan` decides without calling a model — shadow it against your current
routing on real traffic before letting it touch anything.
