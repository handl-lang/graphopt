# Retro — the enumerate arm's attr gap: instantiation reach, fixed

Date: 2026-10-15 (follow-up to `exposed-math.md`)
Context: the exposure retro measured the enumerate arm at **0 fires
across 8 models** (~1,263 candidates) and diagnosed an attr gap —
the enumerator mints `select X` while real exports carry
`select(x, dim=1, index=0)`.  This change fixes the machinery and
re-measures on the same slice.

## The actual defect was two defects

The retro named the attr gap.  Reproducing it showed a second,
equally fatal one underneath:

1. **Attr-free patterns.**  `_term_match` / the compiled `_POp`
   matcher require exact attr-key-set equality —
   `set(term.attrs) == set(pat.attrs)`.  `select X` never matches
   `select(x, dim=1, index=0)`.
2. **Concrete `Var` leaves.**  `enumerate_candidates` instantiated
   metas to `Var("A", (4,4))` inside `Candidate.lhs`, and
   `as_rule()` shipped those leaves into the `Rewrite`.  A `Var` in
   a pattern compiles to `_PLeaf` — keyed by `repr`, matching only
   the enumerator's own instance.  Even an attr-free `add A B`
   candidate could fire nowhere but at its own root — which is why
   `verify_law`/`_cost_delta` "worked" (the own-instance fire) while
   `impact._probe` on real terms saw zero.

So the enumerator was doubly synthetic: evaluated on instances only
it could see.

## The fix — two forms per candidate

`enumerate_candidates` now mints, per rewriting-form pair:

* the **evaluated instance** (`Candidate.lhs`/`rhs`) — `Var`/`Const`
  leaves plus a concrete attr draw from `oracle._attr_options` (the
  same honest domain the view oracle sweeps), shape-informed via
  `typing.shape_of`; a key the domain cannot honestly fill mints the
  bare node (undecidable beats fabricated);
* the **fireable pattern** (`Candidate.rule`, the documented
  `as_rule` seam) — `str` metavar leaves plus one attr metavariable
  per key (`dim="select_dim"`), **shared across every occurrence of
  the op name** so a match binds the real values and the RHS
  re-mints them — the `pipeline._view_node` convention.  Ops absent
  from the LHS keep the instance's concrete attrs: an unbound attr
  metavar would fall back to the literal name string in
  `_term_instantiate` — a minted lie — and concrete is the one sound
  spelling a derive-less rule can produce.

Attr key-sets are data: `attr_keys` maps op → the spellings to mint
(`exposed_discovery` passes the term's own observed sets — `select`
gets `{dim,index}`, `sum` gets `{dim}` *and* `{dim,keepdim}`), and
the product over a pair's ops is capped at `_MAX_SPELLINGS=8`.
Absent a term, `ATTR_SCHEMA` supplies the default (all declared
names, plus the `ATTR_REQUIRED` subset when it differs — the
elided-default shape an export may produce).  Dedup moved to the
*pattern* alpha-key, which also makes the library `relation=` honest
for attr-bearing laws (a minted `select`-naturality can now classify
`inverse` of a shipped rule instead of vacuously `new`).

The matcher itself is **untouched** — no wildcarding, no relaxed
key-sets.  `dim`/`index` stay semantics: bound by the match,
re-minted into the RHS.

## Follow-on defect the fix exposed: unbudgeted candidates

The moment candidates could fire, `proposal._cost_delta`'s
`[*ALL_RULES, rule]` saturation churned to the node cap
(measured >200 s where the never-firing rules ran in 0.1 s).
`rule_budgets` existed for exactly this class — but only covered
EXPANSIVE-tagged *library* rules.  `proposal._rule_budgets` and
`impact._rule_budgets` now budget every non-library rule: a
candidate under test is unbounded-blast-radius by definition.
Factor-left's `_cost_delta` went 200 s+ → 4 s (fixed point at ~1.8k
enodes).

## Measured: `lab/screen/exposed_discovery.py`, same 8-model slice

Per-model enumerate arm (`--evaluate 200`, `impact._probe` on every
new+true candidate):

| model | new+true | candidates firing | total fires |
|---|---|---|---|
| GatedAttentionUnit | 12→12 | 0→3 | 0→3 |
| CosineAttention | 27→70 | 0→12 | 0→68 |
| AdaLNBlock | 20→45 | 0→14 | 0→49 |
| LoRAAdapter | 8→8 | 0→0 | 0→0 |
| CapsuleRouting | 12→11 | 0→1 | 0→2 |
| ChunkedRetention | 61→61 | 0→9 | 0→24 |
| DeepEquilibrium | 1→1 | 0→0 | 0→0 |
| DeltaRuleBlock | 28→24 | 0→5 | 0→52 |
| **total** | | **0→44** | **0→198** |

Verified-passing fires (rule instantiated real attrs, lowered
modules agree): `select∘sigmoid = sigmoid∘select` (4×,
DeltaRuleBlock), `transpose∘relu = relu∘transpose` (GAU),
`reshape∘chunk = chunk∘reshape` (3×, CosineAttention),
`chunk∘unsqueeze = unsqueeze∘chunk` (3×, AdaLNBlock),
`detach_∘contiguous` swap/inverse/absorb (ChunkedRetention),
`unsqueeze∘select` swap (12×, DeltaRuleBlock).  These are the
op-naturality family the census arm knew from
`mixed:*_select_*_w`/`mul_unsqueeze` — the enumerator now reaches
them independently, as the retro predicted it should.

## Honest limits

* **No paying+verified candidates.**  Three AdaLNBlock rows report
  `paid=True` but `verify=FAIL` (`chunk(A, chunks=1) = add(A, 0)` /
  `mul(1, A)` family): the eval draw `chunks=1` makes a
  *conditional* truth (`chunks=1 ⇒ identity`) look unconditional;
  the law is false at `chunks≥2` and the lowered-module verify
  correctly refuses it.  The enumerate arm has no view-oracle /
  guard machinery — conditional truths are caught at verify, not
  proposed with the condition.  Nothing new is shippable; nothing
  shipped.
* **Same-attr sharing is narrower than the truth.**  Attr metavars
  are per-op-*name*, so two occurrences of one op in a pattern must
  bind equal attrs (`select(select x, d, i), d, i` misses
  mixed-attr nestings) — sound, but under-reaches.
* **Eval is one draw.**  The instance checks the first
  `_attr_options` element at the (4,4) leaf shape; a law true only
  off the canonical draw is missed, and (above) a law true only *at*
  it is admitted to the probe stage, where verify disposes.
* **Cost is real now.**  Bounded evals that ran ~0.1 s under
  never-firing rules take ~1–4 s each (LoRAAdapter's 110-eval pool
  took 377 s).  The `--evaluate` budget is the knob.
* **Arity wall stands.**  `concat`/`softmax`/`stack`/`einsum`/
  `zeros`/`arange` report arity `None` and never enter the vocab —
  attr minting can't help ops the grammar can't spell.
* **`forms.py` shares the pre-fix defect.**  `enumerate_form_
  candidates` instantiates the same `_instantiate` Var-leaf,
  attr-free way — same zero-fire exposure, not addressed here.
* `_probe` itself is still unbudgeted (its fire count is the
  reported metric); a self-feeding candidate on a real term is
  bounded only by `_FIRING_NODES`.  Not observed to be slow.

Machine-readable: `enum-attr-gap-before.json`,
`enum-attr-gap-after.json` (this directory).
