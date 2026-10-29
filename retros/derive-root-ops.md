---
title: 'Retro — derive-DSL root ops: closing the measured `sqrt`/`rsqrt` refusal'
type: retro
date: 2026-10-17
---

# Retro — derive-DSL root ops: closing the measured `sqrt`/`rsqrt` refusal

**Scope**: `mlopt_core/laws/cond.py` (the
`dspec` expression bank), `mlopt_discovery/dspec_synth.py` (the
synthesizer's search bank), `lab/probes/weightspace_fold_probe.py`,
`tests/test_cond_laws.py`, `tests/test_discovery_dspec_synth.py`.

**Depends on**: `project/retros/weightspace-fold.md` (the measured
gate-by-gate refusal this closes one named blocker of), plan 0025 §D3
(the weight-space family — derive-or-refuse).

## What the refusal measured, and what changed

The weight-space lane measured the conv+BN fold
`W' = W·g/√(var+eps)`, `b' = (b−μ)·g/√(var+eps) + β` as *structurally
derivable but inexpressible*: the pre-scaled tier fit (`W' = mul(W,
scale)`; `b'` at depth 3), so the missing piece was exactly
`scale = γ·rsqrt(σ²+ε)` — and the DSL had no root op (gate 3:
`dsl_has_sqrt`/`dsl_has_rsqrt` were `False`; float `pow` declines).
This round adds the measured ops, nothing more:

- **`mlopt_core.laws.cond`** — `_DUNOPS` gains `"sqrt": math.sqrt`
  and `"rsqrt": λa. 1/√a`.  Both are float-valued and strict:
  `math.sqrt` raises `ValueError` on a negative operand, `rsqrt`'s
  reciprocal turns a zero operand into `ZeroDivisionError` — both land
  in the `_unop` handler's `(ArithmeticError, TypeError, ValueError)`
  decline set, so the strictness contract (decline, never a fabricated
  complex or `inf`) needs no bespoke handler.  Grammar doc updated
  (`("sqrt"|"rsqrt", e)`); both round-trip through
  `derive_to_data`/`derive_from_data` like every other op.
- **`mlopt_discovery.dspec_synth`** — the *search* bank widens to
  match: `UNOPS` gains `sqrt`/`rsqrt`, `BINOPS` gains `fdiv` (the
  true division the refusal formula spelled literally — `γ/√(σ²+ε)`.
  `fdiv` is load-bearing, not decorative: `g·rsqrt(·)` differs from
  `g/√(·)` in the last ulp, so only the `fdiv`+`sqrt` spelling can
  reproduce a `/`-spelled hook *exactly* under the synth's
  signature-equality gate).  `_vbin`/`_vun` already route through
  `_DBINOPS`/`_DUNOPS`, so no per-op machinery was needed — the ops
  are data, per "laws are data".

`pow` stays int-only (a float exponent would smuggle a root the
strictness contract can't bound — the probe's `pow_float_exp_declines`
still measures `True`).

## The measured delta (probe re-run, fp64)

| gate | before | after |
|---|---|---|
| 3 · DSL spellable | `dsl_has_sqrt`/`rsqrt` = False | **True/True**; `rsqrt_scale_ok` — `mul(γ, rsqrt(add(var, eps)))` evaluates through `eval_derive` |
| 2c · scale fit *(new gate)* | — | depth 2: **no-fit** (17,204 candidates, honest); depth 3: **MINTED** `fdiv(float(float(γ)), sqrt(add(ε, var)))` — `γ/√(ε+σ²)`, exact on all 6 measured channels; sweep truncated at the 60 s budget after 133.9 M pairs but the mint is an exact reproduction |
| 2 · raw fold | `W'`/`b'` = None (depth 2–3) | still **None** — the residual moved: the full fold `fdiv(mul(W,γ), sqrt(add(var,ε)))` sits at depth ≥ 4 over the 7-attr bank, and the streamed level-3 product no longer terminates in budget.  Now an *enumeration-depth* limit, not an op gap |
| 4 · value domain | `Const.value` is `int \| float`; tensor sample raises `TypeError` | **unchanged** — still the dominant blocker |

## What this does and does not buy

- **Closed**: refusal reason 1 of the weight-space retro ("the DSL has
  no square root") — both as spellability (`eval_derive` computes the
  scale) and as *discoverability* (the synthesizer mints the scale
  expr from measured bindings).
- **Open, named**: (a) the derive value domain is still `int | float`
  — a per-channel *tensor* `scale` cannot be a derived value (gate 4
  unchanged); this is the real gate on the conv+BN fold and is an
  architecture question (`bound` carries terms, not tensor values;
  derived attrs must stay JSON-safe; `Const.value` is typed scalar) —
  not a one-line op.  (b) Enumeration depth: the complete fold expr is
  depth 4 over a 7-attr bank; the streamed synthesizer found the scale
  subexpr but the full-expr sweep is budget-bound.  (c) Content, not
  capability: no law LHS names `batch_norm` (probe gate 5) and
  `batch_norm` is unpriced in `_OP_FLOPS` (gate 6).
- Also fixed en passant: the probe's `_load_audit` resolved
  `dspec_audit.py` relative to `lab/probes/` (stale `tools/`-era path);
  it now resolves `lab/audit/dspec_audit.py`.

## Gates

- `pytest tests/test_cond_laws.py tests/test_discovery_dspec_synth.py
  tests/test_discovery_vocab.py tests/test_derive_laws.py
  tests/test_law_serialize.py` — **214 passed** (incl. new
  `TestDeriveRootOps` ×4 and `TestRootOps` ×3: eval, decline semantics
  — negative/zero/non-numeric — JSON round-trip, bank membership, and
  two measured mints: `rsqrt(add(var,eps))` and `fdiv`/`sqrt` fits
  verified through `eval_derive` on every bound).
- `pytest tests/test_laws_structure.py tests/test_property_laws.py
  tests/test_rules.py tests/test_laws_rewrite_edges.py` — 51 passed
  (the widened bank disturbs no structural pin; `pow` still refuses).
- `ruff check` / `ruff format --check` on the touched shipped files —
  clean (the two `tests/` files carry the repo's documented
  pre-existing format drift; the additions themselves are clean).
- `ty check` — 0 errors.  `lint-imports` — 4 contracts kept (`math` is
  stdlib; core stays torch-free).
- `vulture` — one finding, **pre-existing and not mine**:
  `tests/test_form_propose.py:142` (an `if False else` ternary,
  committed in `fb34e18` — the state-reindex agent's file, left for
  them).
