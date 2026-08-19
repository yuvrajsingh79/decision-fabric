# Decision Fabric

**A knowledge graph that decides which Claude model — and which request config — a
query actually needs, before the query is sent.**

Enterprise LLM spend is rarely wasted on volume. It is wasted on *uniformity*:
one flagship model at default settings answering "classify this ticket" and
"prove this algorithm terminates" with the same machinery. Decision Fabric puts
a reasoning layer in front of that call. It derives what a request genuinely
demands, finds the cheapest configuration that meets it, and learns from what
comes back.

On the 30-query enterprise mix in `demo/`, it routes for **$3.72 against a
$5.46 flagship-only baseline — 31.8% saved** — while deliberately spending
*more* than baseline on the three hardest task types. That asymmetry is the
point, and it is discussed honestly in [Findings](#findings-what-the-poc-actually-showed).

---

## The idea in one picture

```
query ──▶ PERCEIVE ──▶ GROUND ──▶ CONSTRAIN ──▶ SCORE ──▶ CONFIGURE ──▶ EXECUTE ──▶ VERIFY ──▶ LEARN
          features     bind to     hard limits   cheapest   model +       Claude      good      write
          from text    the graph   (ctx, SLO)    that       effort +                  enough?   back to
                                                 clears     cache +                             the graph
                                                 the bar    batch …                       │
                                                                                          └──▶ escalate
```

Nothing in that pipeline is an `if model == "..."` branch. Requirements are
*derived* by propagating edges; candidates are whatever the graph says provides
enough; and the numbers on those edges move as evidence arrives. Change the
graph, and you have a different router without touching a line of Python.

---

## What the graph actually contains

| Node | Count | What it is |
|---|---|---|
| `Capability` | 12 | The axes models differ on — `deep_reasoning`, `long_context`, `structured_extraction`, `domain_precision`, … |
| `TaskType` | 16 | `classification`, `rag_qa`, `debugging`, `math_proof`, `agentic_multistep`, … |
| `Domain` | 7 | `legal`, `finance`, `healthcare`, `security`, … |
| `Signal` | 8 | Query-level complexity markers: `adversarial`, `correctness_critical`, `quantitative`, … |
| `Model` | 4 | Fable 5, Opus 5, Sonnet 5, Haiku 4.5 — with pricing, context window, and **config surface** |
| `Policy` | 4 | `economy` / `balanced` / `quality` / `critical` |
| `Slo`, `OutputClass` | 9 | Latency contracts and expected answer lengths |

| Edge | Count | Meaning |
|---|---|---|
| `REQUIRES {min_level}` | 37 | `TaskType → Capability`: the bar a model must clear |
| `PROVIDES {level, seed_level}` | 48 | `Model → Capability`: what a model brings. `level` moves with evidence; `seed_level` never does |
| `ELEVATES {delta}` | 28 | `Domain\|Signal → Capability`: raises the bar for this specific request |
| `ESCALATES_TO` | 3 | The cascade ladder |
| `PRODUCES`, `GOVERNS` | 19 | Task → output size; policy floors by domain |

Inspect it live: `python -m decision_fabric.cli graph --dot | dot -Tsvg > fabric.svg`

---

## How a decision gets made

Take: *"Why is my checkout service throwing intermittent 502s in production?"*

**1. Perceive.** Free regex + token count give `task=debugging`, `domain=general`,
signals `adversarial` (`intermittent`) and `correctness_critical` (`production`),
confidence 0.90. Confidence is the gate: **only below 0.55 does the fabric pay
for an LLM classifier call.** A router that classifies every query with an LLM
has already spent part of what it is trying to save.

**2–3. Ground and derive.** Walk `REQUIRES` from `task:debugging`, then apply
every `ELEVATES` that fired:

```
task:debugging  -REQUIRES 0.84-> cap:deep_reasoning
signal:adversarial          -ELEVATES +0.12-> 0.84 -> 0.86
signal:correctness_critical -ELEVATES +0.05-> 0.86 -> 0.87
```

Elevation uses **diminishing returns** (`level + delta × (1 − level)`), not
addition. Additive stacking made three signals demand 1.14, which no model can
meet, so every multi-signal query pinned to the top of the ladder — the exact
spend the fabric exists to prevent.

**4. Constrain and score.** Every model is checked against the derived bars,
plus three *inviolable* constraints — context window, latency SLO, and data
retention (Fable 5 is unavailable under zero data retention, so `--zdr` removes
it from the pool rather than letting the request 400 at call time):

```
model                 quality    est cost    value  status
claude-haiku-4-5        0.554   $0.040517       14  below bar on deep_reasoning by 0.35
claude-sonnet-5         0.809   $0.081034       10  below bar on deep_reasoning by 0.07
claude-opus-5           0.930   $0.202585        5  SELECTED (primary)
claude-fable-5          0.967   $0.405170        2  relative latency 1.00 exceeds SLO ceiling 0.80
```

The rule is *cheapest model that clears the bar* — not "best value score".
Quality is a constraint to satisfy, not a quantity to trade against cost.

**5. Configure.** Model choice is only half the saving. The other half:

```
PLAN  claude-opus-5 effort=xhigh thinking=adaptive max_tokens=11340
      - required deep_reasoning=0.87 -> effort 'xhigh'
      - adaptive thinking + effort (disabled-thinking avoided by policy)
      - max_tokens raised to 11340 — effort 'xhigh' bills ~4.5x output and
        thinking counts against the ceiling
```

**Each model's config surface is a graph property, so the planner cannot emit a
knob the model rejects.** This is not cosmetic — these are 400s, not degraded
answers:

- `claude-fable-5` — thinking is always on; **any** explicit `thinking` config is rejected. Depth is controlled by `effort` alone.
- `claude-haiku-4-5` — has no `effort` knob at all; it takes legacy `thinking.budget_tokens`, which must be less than `max_tokens`.
- `claude-sonnet-5` / `claude-opus-5` — `thinking: {type: "adaptive"}` plus `effort`; `temperature` is rejected.
- `claude-opus-5` — the only fast-mode-capable model here, and fast mode cannot combine with the Batch API.

The planner also decides **prompt caching** (only above the ~1024-token minimum
cacheable prefix — below it, `cache_control` buys you the 1.25× write premium
and never hits), **Batch API** (50% off when the SLO tolerates async), and
**streaming** (above 16K `max_tokens`, to avoid HTTP timeouts).

**6–7. Execute and verify.** The answer is graded — free heuristics (truncation,
refusal, punting, stubs, task-shaped length floors), or a Haiku judge with a
strict JSON schema under stricter policies. Failure escalates one rung.

**8. Learn.** The verdict is written back as graph evidence. See
[The learning loop](#the-learning-loop).

---

## The cascade is EV-gated, not a reflex

"Try cheap first" is only cheaper when the retry math works. Before probing, the
fabric computes:

```
CASCADE ON: probe $0.000836 + verify $0.000960 + 7% x $0.002090 = $0.001948
            vs $0.002090 direct
```

Note how thin that margin is: the verification pass costs *more than the probe
itself*, and the cascade wins by 7%. On a shorter task it would lose, and the
fabric would say `CASCADE OFF` with the arithmetic attached. Cascading by
reflex, without counting the check, is how "try cheap first" quietly costs more
than it saves.

`p_fail` is a real probability — `P(quality < accept_threshold)` under a normal
around the model's expected quality — not a hand-wave. If the expected cascade
cost isn't below the direct cost, the fabric routes straight to the primary and
says so in the trace.

---

## The learning loop

Two channels, both landing on graph edges rather than in a hidden model:

1. **Beta posteriors per `(model, task_type)`**, anchored on the seed prior:
   `Beta(seed × k, (1−seed) × k)`. With zero observations the posterior *is* the
   seed, and the first observation nudges rather than overwrites. A flat
   `Beta(1,1)` would drop a 0.94-capability model to 0.67 after a single
   success — worse than not learning at all.

2. **Capability drift**, pooled across every task that requires a capability and
   weighted by both evidence volume and how much each task leans on it. Updating
   from one task in isolation meant whichever task ran last won, and levels
   oscillated instead of converging.

Because capabilities are shared, evidence transfers: a model proving itself on
`code_gen` raises its `deep_reasoning` standing for `debugging` too.

Watch it move: `python demo/run_demo.py --rounds 3`

---

## Findings: what the PoC actually showed

Three results worth more than the headline number.

### 1. Learning must not touch the eligibility gate

The first working version let evidence move capability levels, which moved
*eligibility*. Cheap models that failed a few deliberately-cheap probes were
struck from the candidate pool, so everything fell through to the flagship.
Measured over five rounds of the same corpus:

| Round | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Saved | **+28%** | −26% | −26% | −26% | −26% |

The learning loop was optimising quality while *destroying* the cost objective.
The fix is a separation of concerns now enforced by a test
(`test_learning_cannot_revoke_eligibility`): **eligibility is judged against the
immutable `seed_level`; evidence only moves `expected_quality`.** Same
adaptation — a model that keeps failing stops being probed because its EV gate
turns negative — but reached by lowering spend instead of raising it.

### 2. Selection floor and acceptance bar must be different numbers

Using one threshold for "how much headroom before I route here" and "is this
answer shippable" makes any model sitting near it a coin flip; those coin-flip
failures then push it below the bar permanently. Splitting `quality_floor` from
`accept_threshold` per policy stabilised the loop:

| Round | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Saved | +31.8% | +31.8% | +31.8% | +31.8% | +31.8% | +31.8% |
| Cascades | 5 | 2 | 0 | 0 | 0 | 0 |

**Cascades decaying to zero at constant cost is convergence, not failure.** The
cascade is a hedge against uncertainty; once evidence promotes the cheap model
to primary outright, the probe-and-verify overhead disappears.

### 3. The regex layer is not a classifier, and measuring it proved it

Held-out measurement, three independent sets, no tuning between seal and score:

| Set | Status | Raw heuristic accuracy |
|---|---|---|
| `demo/queries.jsonl` | patterns tuned against it | 100% |
| `eval/dev.jsonl` (60 q) | held out, then used for tuning | 31.7% |
| `eval/dev2.jsonl` (91 q) | held out, measured once, then retired | 14.3% |
| `eval/test.jsonl` (75 q) | **sealed** | **16.0%** |

The 100% was curve-fitting. On natural enterprise phrasing the regex layer
classifies at ~15%, with 11 of 16 task types at zero recall. No quantity of
additional patterns fixes a design that requires anticipating phrasing.

So the architecture changed rather than the regexes. **The regex layer is a
precision fast-path, not a classifier**; the LLM is the classifier. And the
fast-path's threshold is no longer a hand-picked number — a fixed 0.55 cut-off
was letting through confidently-wrong classifications, which are the only
classification errors that reach production unchecked. The gate now consults
`config/calibration.yaml`, generated by `eval/calibrate.py`, and commits **only
in confidence bands with measured precision >= 0.90 over >= 5 observations**.
Unmeasured bands always escalate: absence of evidence is not evidence of
reliability.

Measured on the sealed set, with the calibrated gate:

| | Share | Accuracy |
|---|---|---|
| commit — skip the LLM call | 3% | **100.0%** |
| escalate — pay for classification | 97% | 13.7% |
| **silent errors** | | **0** |

Zero confidently-wrong classifications, enforced by `tests/test_gate.py`. The
fast-path is now almost worthless as a *cost* optimisation — and that is the
correct outcome, because it is affordable: at $0.00072 a call, classification
overhead is ~1.6% of the average per-query routing saving. Paying Haiku to
classify is two orders of magnitude cheaper than one misrouted Opus call.

Evaluation discipline (train/dev/test, the content seal, and the rule that
reading a `--test` result retires it) is documented in `eval/README.md`.

**Measured live** (sealed set, Haiku 4.5 classifier, one call per query):

| Metric | Result |
|---|---|
| LLM classifier accuracy | **84.0%** (63/75) |
| Cost | $0.0603 total, **$0.000803/query** |
| Misclassifications that changed the routed model | **5/75 = 6.7%** |
| Misclassifications that *under*-served (quality risk) | **3/75 = 4.0%** |

Classification accuracy is the wrong headline. 7 of the 12 errors routed to the
identical model — `summarization_long` vs `data_analysis` disagreements all land
on Sonnet 5 either way, because the graph derives requirements from
capabilities, not from task labels. **A task-type error only matters when it
crosses a capability boundary.** The number to govern is the 4.0% that
under-served, not the 16% that missed a label.

Weakest classes are `rag_qa` (40%) and `summarization_long` (25%), and several
of those errors are arguable labelling rather than model failure — "what is this
Polish complaint about" is defensibly translation *or* comprehension. The labels
are the author's; they have not been adjudicated by a second annotator, and the
84% is not adjusted for that.

**Config surface verified live.** All 20 emitted configs (4 models x 5 effort
rungs) were accepted by the API for $0.0075 — Haiku 4.5 with legacy
`budget_tokens` and no `effort`, Sonnet/Opus 5 with `adaptive` + `effort`,
Fable 5 with `effort` + refusal-fallback betas and no `thinking` key. Reproduce
with `python demo/live_check.py`.

> **Still unmeasured:** end-to-end routed answer *quality*. The fabric picks a
> model and explains why, and that decision is now verified as valid and
> mostly-correct — but nobody has graded whether the cheaper model's answers are
> actually good enough. The verifier heuristics are the weakest link in that
> chain and have never been scored against human judgement.

### 4. The router is not a discount machine

Savings are not uniform, and pretending otherwise would be dishonest:

| | Task types | Result |
|---|---|---|
| **Saves heavily** | `summarization_long`, `rag_qa`, `extraction`, `classification`, `translation` | 90–99% — a long-context summarisation went from $1.04 to $0.014 via a cheaper model, caching, and batch |
| **Roughly neutral** | `debugging`, `code_review` | ±3% — right model already, small effort win |
| **Deliberately spends more** | `math_proof` (−357%), `architecture_design` (−40%), `agentic_multistep` (−39%) | The graph says these need `xhigh`/`max` effort, which bills 4.5–7× output |

The net is positive because the cheap majority dominates the volume. If you want
the tail capped, that is a policy decision (`effort_bias`), not a router
decision — and it is one line of YAML.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# One decision, fully explained. No API key needed.
PYTHONPATH=src python -m decision_fabric.cli --dry-run route \
  "why is my checkout service throwing intermittent 502s in production?" --trace

# The whole enterprise mix, with a savings report
python demo/run_demo.py

# Watch the graph adapt over repeated rounds
python demo/run_demo.py --rounds 3

# Inspect the fabric and everything it has learned
PYTHONPATH=src python -m decision_fabric.cli graph
PYTHONPATH=src python -m decision_fabric.cli graph --dot | dot -Tsvg > fabric.svg

PYTHONPATH=src python -m pytest tests/ -q     # 30 tests
```

**Live mode.** Export `ANTHROPIC_API_KEY` (or run `ant auth login` — the SDK
picks up the profile with no env var) and pass `--live`. Nothing else changes;
the plan is identical. Without credentials everything runs in a deterministic
simulation so the full pipeline — cascade, verification, escalation, learning —
is demonstrable and testable at zero cost.

**As a service:**

```bash
pip install -r requirements-api.txt
PYTHONPATH=src uvicorn decision_fabric.api:app --reload
```

`POST /plan` is the one to start with — it decides without calling a model, so
you can shadow it against your current routing on real traffic and measure the
delta before it touches anything.

---

## Layout

```
config/
  graph.yaml      capabilities, task types, domains, signals, SLOs, output classes
  models.yaml     model catalog: pricing, context, config surface, capability priors
  policies.yaml   budget policies, domain floors, learning parameters
src/decision_fabric/
  ontology.py       node/edge types; the diminishing-returns elevation rule
  graph.py          NetworkX MultiDiGraph + typed traversals (swap in Neo4j here)
  seed.py           YAML -> graph; time-boxed intro pricing
  features.py       stage 1 — free heuristic extraction
  classifier.py     stage 1b — LLM refinement, only when confidence is low
  reasoner.py       stages 2-4 — requirement derivation, scoring, cascade EV gate
  config_planner.py stage 5 — per-model-valid request construction
  executor.py       stage 6 — Anthropic SDK call, or deterministic simulation
  verifier.py       stage 7 — heuristic and Haiku-judge acceptance
  learning.py       stage 8 — seed-anchored posteriors, pooled capability drift
  router.py         the pipeline
  pricing.py        cost projection and actuals (cache, batch, effort, fast mode)
  telemetry.py      SQLite audit trail + savings reporting
  explain.py        human-readable decision reports
  cli.py / api.py   CLI and FastAPI surfaces
demo/               30-query enterprise corpus + runner
tests/              30 tests, focused on where a bug costs money
```

---

## Honest limitations

This is a proof of concept. What it does not yet do:

- **The capability levels in `models.yaml` are hand-authored seed priors, not
  benchmark results.** They are marked `provenance: seed-prior` in the file. The
  learning loop is what makes this survivable — it converges on your workload —
  but a production deployment should seed from real evals.
- **The baseline is a projection, not a measurement.** The flagship model is not
  actually run in parallel; baseline output length is the task's expected length
  at default effort. Real A/B via `POST /plan` in shadow mode is the honest way
  to validate the savings claim.
- **Dry-run token counts are simulated.** Live mode uses real `count_tokens` and
  real `usage`; the demo numbers above are simulated and internally consistent,
  not measured against the API.
- **The verifier is the weakest link.** Heuristics catch truncation, refusal and
  punting well; they do not catch confident, fluent wrongness. The Haiku judge
  helps, and `critical` policy declines to cascade at all — but a cascade is only
  ever as trustworthy as its check.
- **Single-turn only.** No conversation state, no tool-use loops, no multi-turn
  cache-prefix reuse across a session.
- **In-memory graph.** Fine to ~10⁴ nodes. `KnowledgeGraph` is a thin interface
  precisely so Neo4j or an RDF store can replace it without touching the reasoner.
