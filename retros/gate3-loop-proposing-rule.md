# Retro — gate 3: the `loop`-proposing rule (a measured FAIL)

**Date**: 2026-10-16.
**Commit**: `b7ba9ff` (catopt `main`).
**Decides**: ADR 0008 gate 3 → **FAIL**; ADR 0009 stage 4 is **not**
justified on this evidence.

## What the gate was

ADR 0008's gate 3 asked the one question a presentation cannot answer
about itself: **does the search reach `loop`-headed forms?**  Gate 2
proved the *spelling* works — a hand-written `loop(step, n)` term lowers
and verifies at 1e-7 against `zoo._ChunkedRetention`.  Gate 3 asked
whether *search*, not the author, gets there.  The 2026-10-15 run
returned PARTIAL: the extracted winner reaches the algorithmic content
(a chained shared-state fold, `reads=2 chained=True`, fold-chain depth 2
against the spelled 3) and projects through the bridge EQUAL — but it is
never `loop`-**headed**, because nothing in the search can *introduce* a
`loop`.  The ADR's own words: *"the fold must be proposed by search,
never recovered by the bridge."*

So the missing piece was a **proposing rule** — a rewrite whose
right-hand side introduces `loop`.

## What was built

`lab/bridge/loop_propose.py` (+ `--propose-loop` in
`lab/bridge/combinator_probe.py`; tests in `tests/test_loop_propose.py`).

The rule lives over the **combinator presentation**, which is where
`Loop` actually lives (option (b) of the brief).  It is

```
unroll(step, n)  →  Loop(step, n)
```

* `unroll(step, n)` is the presentation's *spelled* bounded scan: the
  **same** `step` morphism applied `n` times, with the wire bookkeeping
  made explicit — a `perm` before each step, one closing `perm`.  It is
  a genuine, regular structure (one shared `step` subterm), not a
  hand-picked pattern.
* The rule recovers `(step, n)` structurally (`n ≥ 2`; `unroll(step, 1)`
  is just `step`, so allowing it would make every morphism a "fold") and
  rewrites to the combinator.
* **Soundness**: the tests prove `unroll(step, n)` and `Loop(step, n)`
  denote the same morphism — `comb_to_dag` expansion (i.e. `_apply`)
  agrees for `n = 1..4`, and the arities match (`S ⊗ X^n → S ⊗ Y^n`).

The "search" is `saturate_comb` — a deliberately small term-level
rewrite closure (apply every rule at every node to a fixpoint, dedupe,
extract the cheapest by node count).  No e-graph congruence: the point
is reachability + selection, not speed.  A real combinator search is
ADR-0009 stage-4 work; this module measures whether that work has a
target at all.

## The commands and the numbers

```sh
# gate 3, unchanged protocol, rule armed
.venv/bin/python lab/bridge/combinator_probe.py --gate2 --extracted --propose-loop
```

```
extracted term: dag=139  reads=2 chained=True
projects: round-trip EQUAL  morph=1590 dup=92 copies=184
fold-chain depth: spelled=3  extract=2
the presentation's spelled scan is in the search's root e-class: False
-- gate 3: the `loop`-proposing rule --
   loop-propose: winner morph=1590 spelled-scan subterms (n>=2) found=0 ns=[]
      saturation: terms=1 fired={'loop_propose': 0, 'loop_expand': 0}
        loop-headed reachable=False  best_is_loop=False  best_cost=1590
   control (authored spelled scan): spelled-scan subterms=1 ns=[4]
      fired={'loop_propose': 1, 'loop_expand': 0}  loop-headed reachable=True
      best_is_loop=True  best_cost=69 (spelled 285)
```

| seed | spelled-scan subterms (n≥2) | rule fires | loop-headed reachable | best |
|---|---|---|---|---|
| **winner** (the gate) | **0** | **0×** | **False** | flat compilation, 1590 |
| control (authored `unroll(step,4)`) | 1 (n=4) | 1× | True | `Loop`, 69 |

## Verdict — FAIL, against the ADR's criterion

> **FAIL** = the winner is the spelled/unrolled form again → search did
> not reach `loop`.

The winner is the spelled/unrolled form again.  The rule fires **0×** on
it and the closure never reaches a `loop`.  Two independent
measurements agree:

1. **The rule is well-formed.**  On the authored spelled scan it fires
   and the cheapest reachable term *is* `Loop`-headed (69 nodes against
   the spelled 285).  This is the control — it separates "the rewrite
   works" from "the search reaches it".  It is not the gate.
2. **The search never reaches it.**  The winner's projection is a flat
   DAG compilation with **zero** `unroll(step, n≥2)` subterms.
   Independently, the presentation's spelled scan (the gate-2 program)
   lands in a **different e-class** from the root — the saturation
   simply never derives that spelling.

## Why (the useful part)

The reason is measured and is deeper than "the rule was too narrow":

* **The export's unrolled recurrence is *parametrically* regular but not
  *syntactically* regular.**  Each block's step differs by the block
  index (`split(…, index=i)`) and the reassociation re-spells the decay
  rate (`xi_C` outside, `xi_1` inside), so the K steps are K *distinct*
  nodes.  There is no repeated `step` subterm to bind, and no fixed
  syntactic pattern matches the chain.
* **The presentation's whole advantage — a *single shared* `step` — is
  exactly what the flat DAG erased.**  `unroll(step, n)` is regular
  *because* the author wrote it with one `step`; the search's term is
  regular only up to the parameter (the block index), and syntactic
  congruence cannot see through a parameter.

So reaching the loop needs **abstraction over the step** — a
form-level / parametric move (recognise K subterms that agree modulo the
index, bind the index, and fold) — not an e-graph rewrite.  The
presentation makes the loop **expressible** (gate 2, PASS); it does not
by itself make it **findable**.

## What it does and does not prove

**Does prove:**

* The proposing rule exists, is sound (`unroll ≡ Loop`), and *works* on
  the presentation's spelled scan (control PASS).
* On the real gate-3 export the search does **not** reach a
  `loop`-headed form, and the presentation's spelled form is not even in
  the search's root e-class.
* Therefore **the bridge is a presentation, not a search space** — the
  ADR's own FAIL branch, stated with numbers.

**Does not prove:**

* That *no* proposing rule could ever work — only that a
  presentation-level syntactic rule cannot, and *why* (parametric, not
  syntactic, regularity).  A form-level proposer is not ruled out; it is
  named as the missing piece.
* Anything about whether a `catopt-kernel` package is a good idea on
  *other* grounds (navigability, the law/carrier split, the ports).  It
  rules out exactly one justification — "the presentation is a discovery
  substrate" — which is the one gate 3 existed to test.

## Consequence for ADR 0009 stage 4

Stage 4 ("promote the calculus to a dimension-named package") was
**gated on gate 3**.  Gate 3 measured negative, so on this evidence
stage 4 is **not** justified by the presentation's discovery value.  This
is the outcome ADR 0009's falsification clause asked to be measured and
written down rather than deferred again.  If stage 4 proceeds, it must
rest on a different argument (and a real form-level proposer is the
piece of work that would re-open the gate).

## Cost

* ~40 s wall for the full `--gate2 --extracted --propose-loop` run
  (gate 2 verify + one 15–16 s staged saturation + the closure); peak
  RSS well under 1 GB — no memory pressure.
* The closure is trivial on the winner (1 term, 0 fires).  The control's
  closure is bounded by `max_terms=4000`; it is not tuned for speed
  because speed is not the question.
* New surface: one lab module (~360 lines), one test file (10 tests),
  ~40 lines of probe wiring.  No shipped-package change, no law-bank
  change, no `catopt_core.ir` node kind (the rule is over the
  presentation, not the IR).

## Gates run

`.venv/bin/ruff check lab tests/test_loop_propose.py`,
`ruff format --check` (clean), `.venv/bin/ty check` (clean),
`.venv/bin/vulture` (clean), `.venv/bin/lint-imports` (4 kept, 0 broken),
`scripts/radon_ratchet.py` (ok), and
`pytest tests/test_combinator_bridge.py tests/test_loop_propose.py`
(22 passed).  The full suite was **not** run (memory/time budget; no
shipped-package change).
