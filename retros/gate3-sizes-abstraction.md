---
title: 'Retro — gate 3: the block width is not the binding cause (a measured FAIL)'
type: retro
date: 2026-10-16
---

# Retro — gate 3: the block width is not the binding cause (a measured FAIL)

**Commit**: `514888c` (catopt `main`).
**Decides**: ADR 0008 gate 3 → still **FAIL**; the first remaining cause
the last retro named (*the two chunk widths / `sizes`*) is **measured**
and is **not** the binding constraint.  The binding constraint is
structural: the cross term is the **unfolded** geometric sum, so the `K`
blocks are not the same step even modulo the width *and* the index.
ADR 0009 stage 4 stays unjustified.

## What the gate was, and what was asked

The state-spelling retro
(`gate3-extraction-state-spelling.md`) closed with two remaining causes:

1. **the two chunk widths** — `sizes=(4, 4, 4, 4, 16, 16, 16)` and
   `sizes=(16, 16, 16, 16)`, both with indices `[0,1,2,3]` — because the
   abstraction's `INDEX_ATTRS` "abstracts `index`/`t`/`offset`/`sign`
   but **not `sizes`**", so the two families could never be one step; and
2. **the abstraction's structural state identification**.

This task attacked cause 1: abstract `sizes`, so a step is parametric in
its block **width** as well as its index.  A negative result is valid;
naming the next cause precisely is the deliverable if it fails.

## What was actually there (a correction to the brief)

`INDEX_ATTRS` was **dead code**.  It is defined and exported in
`lab/bridge/index_abstract.py` and referenced **nowhere** — the move's
`_abstract_block` abstracted only two things: a block's *own* `split`
(replaced wholesale by an input **wire**) and one *state* subterm
(replaced by `@S`).  Replacing the whole split means the index **and**
the width **and** the dim all leave the template at once, so a block's
own width was **already** a parameter (it rides on the wire).  The
retro's claim — that `INDEX_ATTRS` abstracts `index`/`t`/`offset`/`sign`
but not `sizes` — was therefore not describing the code.

What *did* survive into a template was the width on the nodes the move
does **not** abstract: an inherited read of an earlier block (the state's
own construction) and the decay vectors (`decay_vec(…, t=C)`,
`decay(…, t=C)`).  Those carry the per-family width and the per-block
index, and nothing erased them.

## What was built

`lab/bridge/index_abstract.py` — the move now wires `INDEX_ATTRS` in and
adds the width:

* `WIDTH_ATTRS = ("sizes",)`; `PARAM_ATTRS = INDEX_ATTRS + WIDTH_ATTRS`
  — the *parameters* of a step (block index + block width).
* `_param_key(t)` — a template's identity with the param-attr **values**
  erased (every op, child, and other attribute kept, so a genuine
  structural difference still separates two keys).
* `templates_agree_mod(root)` — *are the `K` blocks the same step once
  the index and the width are parameters?*  **Strictly weaker** than
  `templates_agree`; it can only *explain* a negative, never replace one
  (the verdict still requires the exact collapse **and** the round-trip).
* `param_families(root)` — the distinct `(attr, value)` pairs per
  template, so the negative can name the parameter it is up against.
* `AbstractionReport.agree_mod` / `diff_mod` / `param_families`, printed
  by `abstract_index_report`.

`_abstract_blocks`/`_agree` factor the shared abstraction so the exact
and the parametric checks cannot disagree about what a block *is*.
`tests/test_index_abstract.py` grew 5 tests (13 → 18).

**How `sizes` becomes a parameter — the exact answer.**  Two cases:

* A block's **own** `split` → the width rides on the **input wire** the
  move already introduces (the split node is replaced wholesale).  This
  needed no change: a fold whose blocks differ only in their own width is
  *already* exactly regular, and `test_own_split_width_is_already_a_wire`
  pins that.
* A node that **survives** into a template (an inherited read, a
  `decay`/`decay_vec`) → the width is a parameter **by erasure**: the
  attr value is dropped from the template's identity, so the `K` blocks
  are recognised as the same step *up to* the width.  It cannot be
  *folded* — the IR is fixed-shape and has no width parameter — so the
  recognition is reported, never promoted.

## The commands and the numbers

`PYTHONHASHSEED=0` throughout (see the caveats — the saturation is not
reproducible across processes, and a *concurrent* `catopt_core/egraph`
edit moved the enode count during this task; the winner's **cost** is
stable).

```sh
PYTHONHASHSEED=0 .venv/bin/python lab/bridge/combinator_probe.py \
    --gate2 --extracted --propose-loop --abstract-index                  # greedy
PYTHONHASHSEED=0 .venv/bin/python lab/bridge/combinator_probe.py \
    --gate2 --extracted --propose-loop --abstract-index --sharing-extract # + fix arm
```

| | before (retro 3) | after (this task) |
|---|---|---|
| extracted cost (FLOPs) | 455,752 / 432,696 | **455,752 / 451,128** (unchanged; ~~432,696~~ superseded 2026-10-15) |
| extracted dag | 139 / 133 | 139 / 133 |
| bridge morph | 1590 / 1509 | 1590 / 1509 |
| bridge round-trip | EQUAL / EQUAL | EQUAL / EQUAL |
| `loop`-propose fires | 0 | 0 |
| **abstraction verdict** | **IRREGULAR** | **IRREGULAR** |
| first difference | `block1/block2/add[1]/mul[1]: op om_decayed vs add` | **same** |
| *modulo width/index* | — | **still distinct — structural** |

New probe output (greedy arm; the sharing arm is identical here):

```
      verdict=IRREGULAR — the 7 blocks do NOT collapse to one shared step
      first difference: block1/block2/add[1]/mul[1]: op om_decayed vs add
      ... and they are NOT the same step even modulo the block width and
      index — the difference is structural, not a width/index parameter
      first difference mod width/index: block1/block2/add[1]/mul[1]: op om_decayed vs add
      param values blocks=[0]: t=4
      param values blocks=[1]: index=0, offset=4, sizes=(4, 4, 4, 4, 16, 16, 16), t=4
      param values blocks=[2]: index=0, index=1, offset=4, sizes=(4, …), t=1, t=4
      param values blocks=[3]: index=0, index=1, index=2, offset=4, sizes=(4, …), t=1, t=4
      param values blocks=[4]: offset=16, t=16
      param values blocks=[5]: index=1, offset=16, sign=-1, sizes=(16, 16, 16, 16), t=1, t=16
      param values blocks=[6]: index=2, offset=16, sign=-1, sizes=(16, 16, 16, 16), t=16
```

### The decisive measurement

Abstracting the width (and `index`/`t`/`offset`/`sign`) **does not
collapse the templates**:

| | exact distinct | distinct **modulo all param attrs** |
|---|---|---|
| greedy winner | 7 | **7** |
| sharing-aware winner | 7 | **7** |

So the width was **never** the binding constraint: the blocks differ
*structurally*, not in a width/index parameter.  The first such
difference is at the **cross position** (`add[1]/mul[1]`): block 1
spells `mul(ξ, om_decayed(…))` — a **one-term** cross — while block 2
spells `mul(ξ, add(mul(ξ₁, om_decayed(…)), om_decayed(…)))` — a
**two-term** cross.  The blocks differ in the **arity of the accumulated
geometric sum**.

### The controls (the new code does what it claims)

Synthetic folds, `tests/test_index_abstract.py`:

| control | `agree` | `agree_mod` | verdict |
|---|---|---|---|
| own-split width varies `(4,16,16,4)` | **True** | True | **regular**, round-trip True |
| width on a *surviving* node varies | False | **True** | irregular (never a pass) |
| a *structural* difference (block 2 extra factor) | False | **False** | irregular, `diff_mod` named |

## Verdict — FAIL, against the ADR's criterion

> **FAIL** = the winner is the spelled/unrolled form again → search did
> not reach `loop`.

The winner is still the spelled/unrolled form.  The width abstraction
moves **nothing**: same cost, same dag, same morph, same first
difference, same `IRREGULAR` verdict.  The gate stays **FAIL**.

## The next cause, named precisely

`block1/block2/add[1]/mul[1]: op om_decayed vs add`.

It is the **state spelling**, now localised to an exact structural fact:
**block `i`'s cross term is the *unfolded* geometric sum — `i`
accumulated terms — not a single read of a shared state subterm.**  So:

* there is **no state subterm to bind**: block 1's cross reads one term,
  block 2's reads two nested inside an `add`, block 3's reads three.  The
  difference is the **number of accumulated terms**, which no parameter
  (index, width, or any attr) can absorb; and
* the **state-identification limitation is not independent of it** and is
  **not fixable in the move** — measured, not asserted (below).

To make the blocks one step the cross sum must be **factored** into
`q̃ᵢ @ S_{i-1}` — the reassociation the search already has
(`om_shared_mat_reassoc`) — and extraction does **not** prefer it for the
small-chunk family (`C²d < Cd²` at `C=4 < d=16`), which the
state-spelling retro measured and flagged as a *genuine cost optimum*.
So forcing it would be manufacturing a result, and this task does not.

## Is the state-identification limitation fixable in the move?

**No — measured.**  `_state_inputs` picks "the outermost shared subterm
carrying none of the block's own splits"; on the winner it picks a
*component* (η, `unsqueeze(decay_vec(sign=-1))`, or `ξ⊙v`) rather than
the accumulated state.  I tried the strongest relaxation — allow **any**
*shared* candidate, own splits included — and the verdict and the first
difference are **byte-identical** on both arms:

```
current          agree=False agree_mod=False verdict=irregular
                 diff='block1/block2/add[1]/mul[1]: op om_decayed vs add'
relaxed-shared   agree=False agree_mod=False verdict=irregular
                 diff='block1/block2/add[1]/mul[1]: op om_decayed vs add'
```

The reason is structural: for the small-chunk family **there is no shared
state subterm anywhere in the term** — the state `S_{i-1}` is *re-derived*
per block from `k_j`, `v_j`, `η` with the query `q̃ᵢ` folded inside
(`om_decayed(matmul(q̃ᵢ, k_jᵀ), η, v_j)`), so it is q-dependent and cannot
be shared.  No choice of *state subterm* changes the cross term's
**arity**.  The heuristic is therefore not the bug; the spelling is.  It
is left unchanged (a tweak would be churn that buys nothing).

## What it proves, and what it does not

**Proves:**

* **The width is not the binding cause.**  Abstracting `sizes` (and
  `index`/`t`/`offset`/`sign`) leaves **7 distinct** templates on both
  arms — the blocks differ structurally, at the cross position, in the
  arity of the unfolded sum.  The state-spelling retro's cause 1 is
  **retired** as a candidate.
* **The move now answers the width question honestly.**  `agree_mod`
  distinguishes "not the same step" from "not the same step even up to
  the width and index"; the controls show the recognition works
  (surviving-width fold → `agree_mod=True`) and does not over-fire
  (structural fold → `agree_mod=False`).  It is strictly weaker than the
  exact collapse and never becomes a verdict.
* **The state-identification limitation is not fixable in the move** —
  the strongest relaxation changes nothing, because there is no shared
  state subterm to find.

**Does not prove:**

* That no move could ever close the gate.  It retires the *width* class
  of fix and shows the *state-identification* class is downstream of the
  spelling; the spelling (factoring the cross sum) is the remaining
  piece — and the cost model does not prefer it for free.
* Anything about `catopt_core.egraph.extract` as shipped — no shipped
  package changed.

## Caveats (read before re-measuring)

* **Seed.**  The staged saturation consumes `rule_budgets` in a
  set-iteration order, so it is not reproducible across processes; every
  number here is `PYTHONHASHSEED=0`.  The winner's *cost* (455,752 /
  ~~432,696~~ **451,128**) and read sequence are stable across seeds.
  (**Corrected 2026-10-15**, `abf6114`: the saturation is now
  deterministic — canonical member order — and the sharing-aware optimum
  is 451,128 in every process; the 432,696 quoted here was a hash-order
  artifact.)
* **A concurrent edit moved the enode count.**  During this task another
  change was in flight in `catopt_core/egraph/{core,extract,types}.py`;
  the staged arm reports **9784** enodes here against the retro's
  **9733**.  Verified not to be this task's doing: the probe without
  `--abstract-index` (which never touches the move) also reports 9784,
  and the winner's cost/dag/morph are unchanged.  Re-measure on a quiet
  tree.

## Cost

* Each gate-3 arm: one 16 s staged saturation; the sharing-aware arm
  adds an 88 s pass (both well under 1 GB RSS).
* New surface: `lab/bridge/index_abstract.py` (+272 / −34 — the
  `_abstract_blocks`/`_agree` refactor plus the new width API), 5 tests.
  **No shipped-package change**, no law-bank change, no `catopt_core`
  change.

## Gates run

`.venv/bin/ruff check lab/bridge/index_abstract.py` and
`ruff format --check` (clean; `tests/` is not format-checked),
`.venv/bin/ty check` (clean), `.venv/bin/vulture` (clean),
`.venv/bin/lint-imports` (4 kept, 0 broken),
`.venv/bin/bandit -c .bandit.yaml -r packages` (0 findings),
`.venv/bin/semgrep --config .semgrep.yml packages` (0 findings),
`scripts/radon_ratchet.py` (ok, 4712 functions), and
`pytest tests/test_index_abstract.py tests/test_loop_propose.py
tests/test_combinator_bridge.py` (40 passed).  The full suite was **not**
run (memory/time budget; no shipped-package change).
