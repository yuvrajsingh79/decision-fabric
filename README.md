<h1 align="center">Decision Fabric</h1>

<p align="center">
  <em>A knowledge graph that decides which Claude model — and which request config — a query actually needs.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/tests-41%20passing-2ea043" alt="41 tests passing">
  <img src="https://img.shields.io/badge/status-proof%20of%20concept-f59e0b" alt="Proof of concept">
  <img src="https://img.shields.io/badge/Claude-Opus%205%20·%20Sonnet%205%20·%20Haiku%204.5-D97757?logo=anthropic&logoColor=white" alt="Claude models">
  <img src="https://img.shields.io/badge/graph-NetworkX-ff6f00" alt="NetworkX">
  <img src="https://img.shields.io/badge/API%20cost%20to%20run-%240.00-3fb950" alt="Free to run">
</p>

---

Enterprise LLM spend is rarely wasted on volume. It is wasted on **uniformity** —
one flagship model at default settings answering *"classify this ticket"* and
*"prove this algorithm terminates"* with the same machinery.

Decision Fabric puts a reasoning layer in front of that call. It works out what
a request genuinely demands, picks the cheapest configuration that meets it,
explains the choice, and learns from the outcome.

**25.3% saved** on a 30-query enterprise mix — while deliberately spending *more*
on the hardest queries, because that is what they needed.

## What it looks like

```console
$ decision-fabric route "review this authentication diff for security holes" \
      --task code_review --policy quality --slo background

QUERY   review this authentication diff for security holes
BOUND   task=code_review  domain=general  slo=background  policy=quality(floor 0.85)

REQUIREMENTS DERIVED FROM THE GRAPH
  code_review            >= 0.82   <- code_review
  deep_reasoning         >= 0.75   <- code_review

CANDIDATES
  model                 quality    est cost    value  status
  claude-haiku-4-5        0.536   $0.027017       20  below bar on code_review by 0.27
  claude-sonnet-5         0.800   $0.054034       15  below bar on code_review by 0.02
  claude-opus-5           0.930   $0.135085        7  SELECTED (primary)
  claude-fable-5          0.969   $0.270170        4  eligible, not cheapest

PLAN    claude-opus-5 effort=high thinking=adaptive max_tokens=8192
        - required deep_reasoning=0.75 -> effort 'high'
        - adaptive thinking + effort (disabled-thinking avoided by policy)
```

Every model considered, what each would cost, which capability bar each cleared
or missed **by how much**, the config chosen and why. Sonnet 5 lost by 0.02 —
and you can see that, rather than being told a model was picked.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

```bash
decision-fabric route "why is my checkout throwing intermittent 502s?" --trace
python demo/run_demo.py                # 30-query mix + savings report
python demo/run_demo.py --rounds 3     # watch the graph adapt
decision-fabric graph                  # the fabric and what it has learned
pytest tests/ -q                       # 41 tests
```

**None of that costs a cent.** The whole system runs in a deterministic
simulation — cascade, verification, escalation and learning included — so you
can evaluate it before pointing it at your account.

To use it for real, authenticate and add `--live`:

```bash
ant auth login          # or: export ANTHROPIC_API_KEY=...
decision-fabric route "your question" --live --show-answer
decision-fabric spend --budget 5       # what you have actually spent
```

## How it works

```
query ──▶ PERCEIVE ──▶ GROUND ──▶ CONSTRAIN ──▶ SCORE ──▶ CONFIGURE ──▶ EXECUTE ──▶ VERIFY ──▶ LEARN
          what kind    bind to    hard limits   cheapest   model +       call        good      write back
          of request   the graph  (ctx, SLO,    that       effort +      Claude      enough?   to the graph
                                  residency)    clears     caching +                    │
                                                the bar    batch …                      └──▶ escalate
```

Nothing in that pipeline is an `if model == "..."` branch. Requirements are
**derived by walking graph edges**, candidates are whatever the graph says
provides enough, and the numbers on those edges move as evidence arrives.
Change the graph and you have a different router without touching Python.

<details>
<summary><b>What the graph contains</b> — 60 nodes, 135 edges</summary>

| Node | Count | What it is |
|---|---|---|
| `Capability` | 12 | The axes models differ on — `deep_reasoning`, `long_context`, `domain_precision`, … |
| `TaskType` | 16 | `classification`, `rag_qa`, `debugging`, `math_proof`, `agentic_multistep`, … |
| `Domain` | 7 | `legal`, `finance`, `healthcare`, `security`, … |
| `Signal` | 8 | Query-level markers: `adversarial`, `correctness_critical`, `quantitative`, … |
| `Model` | 4 | With pricing, context window, and **config surface** |
| `Policy` | 4 | `economy` · `balanced` · `quality` · `critical` |

| Edge | Meaning |
|---|---|
| `REQUIRES {min_level}` | `TaskType → Capability` — the bar a model must clear |
| `PROVIDES {level, seed_level}` | `Model → Capability` — `level` moves with evidence, `seed_level` never does |
| `ELEVATES {delta}` | `Domain\|Signal → Capability` — raises the bar for *this* request |
| `ESCALATES_TO` | The cascade ladder |

Export it: `decision-fabric graph --dot | dot -Tsvg > fabric.svg`

</details>

<details>
<summary><b>How requirements are derived</b> — the part that isn't a lookup table</summary>

For *"why is my checkout service throwing intermittent 502s in production?"*:

```
task:debugging  -REQUIRES 0.84-> cap:deep_reasoning
signal:adversarial          -ELEVATES +0.12-> 0.84 -> 0.86
signal:correctness_critical -ELEVATES +0.05-> 0.86 -> 0.87
```

Same task type, harder instance, different model. Elevation uses diminishing
returns rather than addition, so stacked signals raise the bar without pinning
every multi-signal query to the top of the ladder.

</details>

<details>
<summary><b>Why the config matters as much as the model</b></summary>

Each model's config surface is a graph property, so the planner **cannot emit a
knob a model rejects** — these are 400s, not degraded answers:

- **Fable 5** — thinking is always on; any explicit `thinking` config is rejected. Depth is controlled by `effort` alone.
- **Haiku 4.5** — has no `effort` knob; takes legacy `thinking.budget_tokens`, which must be less than `max_tokens`.
- **Sonnet 5 / Opus 5** — `thinking: {type: "adaptive"}` plus `effort`; `temperature` is rejected.
- **Opus 5** — the only fast-mode-capable model, and fast mode cannot combine with the Batch API.

*Verified live: all 20 emitted configs (4 models × 5 effort rungs) accepted by
the API.* The planner also decides prompt caching, Batch eligibility, and
streaming.

</details>

<details>
<summary><b>The cascade is expected-value gated, not a reflex</b></summary>

"Try cheap first" only saves money when the retry math works, so the fabric
computes it before probing:

```
CASCADE ON: probe $0.000836 + verify $0.000960 + 7% x $0.002090 = $0.001948
            vs $0.002090 direct
```

The verification pass costs *more than the probe itself* and the cascade still
wins by 7%. On a shorter task it would lose, and the fabric says `CASCADE OFF`
with the arithmetic attached.

</details>

<details>
<summary><b>How it learns</b></summary>

Verdicts are written back as graph evidence, not into a hidden model:

1. **Beta posteriors per `(model, task_type)`**, anchored on the seed prior — with no observations the posterior *is* the seed, and evidence nudges rather than overwrites.
2. **Capability drift**, pooled across every task that requires a capability. Because capabilities are shared, evidence transfers: a model proving itself on `code_gen` raises its `deep_reasoning` standing for `debugging` too.

Learning never revokes a model's eligibility — it adjusts scoring, so a model
that keeps failing stops being *probed* rather than being *banned*. See
[docs/FINDINGS.md](docs/FINDINGS.md) for why that distinction is load-bearing.

</details>

## Configuration

Everything routable lives in YAML. No Python changes required.

| File | What you tune |
|---|---|
| `config/models.yaml` | Model catalog: pricing, context window, config surface, capability priors |
| `config/graph.yaml` | Capabilities, task types, domains, complexity signals, SLOs, output sizes |
| `config/policies.yaml` | Budget policies, per-domain floors, learning parameters |
| `config/calibration.yaml` | *Generated* — the classification gate's measured precision map |

**Policies** set how much capability headroom is required before routing:

| Policy | Floor | Cascade | Batch | Use for |
|---|---|---|---|---|
| `economy` | 0.60 | yes | yes | Bulk internal workloads |
| `balanced` | 0.72 | yes | yes | Default knowledge work |
| `quality` | 0.85 | yes | no | Customer-facing output |
| `critical` | 0.93 | no | no | Regulated / irreversible |

Regulated domains force a minimum policy regardless of what the caller asked
for — a `healthcare` query submitted as `economy` is routed as `quality`, and
the trace says so.

## Using it

<details>
<summary><b>CLI</b></summary>

```bash
decision-fabric route "your query" [--policy economy|balanced|quality|critical]
                                   [--slo realtime|interactive|background|batch]
                                   [--domain legal|finance|healthcare|…]
                                   [--task code_review]      # skip classification
                                   [--context-tokens N] [--stable-tokens N]
                                   [--zdr]                   # zero-data-retention org
                                   [--plan-only] [--trace] [--json] [--live]

decision-fabric graph      # nodes, edges, ladder, learned posteriors
decision-fabric costs      # price list: model × effort
decision-fabric spend      # your actual spend
decision-fabric report     # cumulative savings vs baseline
```

Full catalogue, with each command marked free or billable:
**[COMMANDS.md](COMMANDS.md)**

</details>

<details>
<summary><b>Python</b></summary>

```python
from decision_fabric import Router

router = Router()                      # dry-run by default; dry_run=False to spend
decision = router.route(
    "summarise these support transcripts",
    policy="economy",
    latency_slo="batch",
    context_tokens=180_000,
)

print(decision.chosen_model)                        # claude-sonnet-5
print(decision.selection.first_plan.summary())      # …effort=low thinking=adaptive max_tokens=2048 batch
print(decision.saved_pct)                           # 99.67
print(decision.answer)
```

</details>

<details>
<summary><b>HTTP service</b></summary>

```bash
pip install -r requirements-api.txt
uvicorn decision_fabric.api:app --reload                 # simulated (safe default)
DECISION_FABRIC_LIVE=1 uvicorn decision_fabric.api:app   # real calls
```

`POST /plan` decides **without calling a model** — shadow it against your
current routing on real traffic and measure the delta before it touches
anything. `POST /route` executes. `GET /savings` and `GET /graph` report.

</details>

## What it costs to run

Three components can make an API call:

| Component | Model | When | Typical |
|---|---|---|---|
| Classifier | Haiku 4.5 | when the gate can't trust the fast path | $0.0007 |
| **The answer** | the routed model | every route unless `--plan-only` | *varies* |
| Verifier | Haiku 4.5 | `quality` / `critical` policies only | $0.0013 |

The first and third are the router's own overhead — about **1.6% of what it
saves**. Run `decision-fabric costs` for the full model × effort price grid.

> **Cost safety.** Nothing spends money without `--live`, even when credentials
> are present. `pytest` cannot spend at all — an autouse tripwire blocks live
> client construction, so the suite proves it rather than promising it. Every
> live call, from any entry point, lands in one ledger: `decision-fabric spend`.

## Measured results

| | Result | How |
|---|---|---|
| Savings on the 30-query mix | **25.3%** | simulated, internally consistent |
| Config surface conformance | **20/20 accepted** | live, $0.0075 |
| LLM classifier accuracy | **84.0%** | live, sealed held-out set, $0.060 |
| Misclassifications that changed the routed model | **6.7%** | a wrong label only matters when it crosses a capability boundary |
| Cheap-tier live run | **$0.0203** for 7 queries | live |

Evaluation uses a train/dev/test split with a content-sealed test set; reading a
test result retires it. See [eval/README.md](eval/README.md) for the discipline
and [docs/FINDINGS.md](docs/FINDINGS.md) for what the measurements revealed.

## Limitations

This is a proof of concept. What it does **not** do yet:

- **Capability levels are hand-authored seed priors**, not benchmark results — marked `provenance: seed-prior` in `config/models.yaml`. The learning loop converges them onto your workload, but a production deployment should seed from real evals.
- **The Batch API is planned for but never called.** The planner marks requests batch-eligible; execution is always synchronous. The 50% batch saving is unrealised in live mode, and the cost model no longer claims it. This is the largest unclaimed saving for bulk work.
- **Routing overhead exceeds the saving on trivial queries.** A four-token answer costs more to route than to send straight to the flagship. There is no short-circuit for that yet.
- **The verifier is the weakest link.** Heuristics catch truncation, refusal and punting well; they do not catch confident, fluent wrongness. Answer quality has never been graded against human judgement.
- **Single-turn only.** No conversation state, tool-use loops, or cross-session cache reuse.
- **In-memory graph.** Fine to ~10⁴ nodes. `KnowledgeGraph` is a thin interface so Neo4j or an RDF store can replace it without touching the reasoner.

## Layout

```
config/          models, ontology, policies, generated calibration  (all YAML)
src/decision_fabric/
  graph.py       NetworkX MultiDiGraph + typed traversals
  reasoner.py    requirement derivation, scoring, cascade EV gate
  config_planner.py  per-model-valid request construction
  executor.py    Anthropic SDK call, or deterministic simulation
  verifier.py    heuristic and LLM-judge acceptance
  learning.py    seed-anchored posteriors, pooled capability drift
  router.py      the pipeline
demo/            corpora, runner, live conformance check
eval/            train/dev/test discipline, sealed set, calibration
docs/FINDINGS.md engineering notes and measurements
COMMANDS.md      every command, marked free or billable
```
