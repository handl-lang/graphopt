# Retro — tensor-valued derives (ADR 0012): the `$leaf:` channel lands

**Date**: 2026-10-18.  **ADR**: `project/adrs/0012-tensor-derives.md`.
**Depends on**: `project/retros/weightspace-fold.md` (the measured
gate-by-gate refusal — gate 4, the last *capability* blocker),
`project/retros/derive-root-ops.md` (the residual after
`sqrt`/`rsqrt`), ADR 0004 (the `dspec` DSL extended).

## What the refusal measured, and what changed

The weight-space lane measured the conv+BN fold as *structurally
derivable but domain-blocked*: the synthesizer fit `W' = W·scale` and
`b' = (b−μ)·scale + β` given a `scale` leaf, the DSL learned to spell
`scale = γ·rsqrt(σ²+ε)` over scalars — and the value domain `int |
float` refused the per-channel *tensor* `scale` with a measured
`TypeError: unhashable type: 'numpy.ndarray'` in the synthesizer's
fingerprinting (gate 4).  This round lands the seam ADR 0012 chooses,
nothing more:

- **`mlopt_core.laws.cond`** — a pushed **tensor algebra**,
  `register_tensor_algebra(source=, mint=, ops=, key=)`: core imports
  no array library, so a tensor value is an *opaque payload* the
  adapter's registrations create, transform and name — the same sink
  direction as `mlopt_core.ops`' binding tables and
  `register_value_shape`.  Two new expr ops: `("tval", T)` (a bound
  leaf's tensor value through `source`) and `("tapp", name, e…)` (the
  generic apply into the pushed `ops` table — `reshape`, `exp`, … —
  so the vocabulary boundary is the table, not the grammar).  The
  `_DUNOPS`/`_DBINOPS` handlers gain a tensor tier: scalars keep
  today's semantics; a non-scalar operand dispatches to `ops[op]`
  when registered, else to the Python operator (which array backends
  already overload — only `math.*`-backed `sqrt`/`rsqrt` *need* the
  table).  And the **`$leaf:NAME` spec key**: the expr's value is
  packaged as a *term* — `Const(v)` for a scalar, `Param(name,
  shape)` through `mint` for a tensor — emitted under the bare
  metavar key, so the RHS pattern's term metavar resolves it at
  instantiate time.  `mint`'s name must be deterministic in content
  (certificate replay re-runs the derive).
- **`egraph/core.py` `apply_rule`** — non-`$attr:` keys in a derive's
  output are *term* bindings: interned via `add_term` so `subst`
  stays uniform (metavar → eid) for the recorded `ProofEdge`.
  Term-level sites (`instantiate_pattern`, `_term_instantiate`,
  `apply_rewrite_at`) needed no change — `subst[name]` already drops
  a bound term in place.
- **`mlopt_core.cost.basic`** — `batch_norm` joins `_OP_FLOPS` at 4
  (sub + mul + mul + add ≈ four elementwise passes over `n_out`; the
  measured gate-6 gap).  `group_norm`/`instance_norm` stay unpriced —
  unmeasured here.
- **`mlopt_discovery.dspec_synth`** — fingerprints box every value
  through `_vkey` (scalars by value, sequences elementwise, payloads
  through the algebra's `key` — or by identity when none is
  registered, so a tensor fit without `key` refuses honestly rather
  than crashing); `("tval", T)` joins the leaf bank when a `source`
  is registered; `tval` counts as reading bound for the no-constant
  guard; `_vbin`/`_vun` mirror the tensor dispatch so value-level
  signatures agree with `eval_derive`.

## The measured delta (probe gate 4 re-run)

| signal | before | after |
|---|---|---|
| `tensor_sample_error` | `TypeError: unhashable type: 'numpy.ndarray'` | **None** |
| scale as a `$leaf:` derive | unspellable | **`scale_mints_as_leaf: True`** — `scale__aecaec8b1d30` mints a `Param` through `mint`, value-exact against `γ/√(σ²+ε)` |
| `tensor_synth_fit` | crash | **`{'Wn': ('mul', ('attr','g'), ('tval','w'))}`** — the synthesizer mints a tensor derive |
| `batch_norm` price | absent (default) | **`4·n_out`** |
| `batch_norm` in a law LHS | absent | **unchanged** — content, not capability (the law is derivable next; the no-hardcoding rule holds) |

End-to-end verified at both instantiate sites: an e-graph
`apply_rule` fires a `$leaf:` rule and lands the minted `Param`'s
enode in the class, and term-level `apply_rewrite_at` replays the
derive **deterministically** (same minted name twice — the replay
contract).

## What this does and does not buy

- **Closed**: weight-space refusal reason 2 (the `int|float` value
  domain) and reason 4 (`batch_norm` unpriced).  The fold's remaining
  blockers are *content*: a `batch_norm` LHS law — derivable now —
  and synth enumeration depth ≥4 over the 7-attr bank (a budget
  question, `derive-root-ops.md` item b).
- **Known limits** (ADR 0012 states them): `meta.synthesize_rules`
  still requires RHS *term* metavars LHS-bound, so a `$leaf:` rule
  cannot be auto-composed (the `@i:` placeholder machinery is
  attr-only); `mint`'s determinism is a documented contract, not an
  enforced one; a `$leaf:` value that is neither scalar nor mintable
  declines — no third landing zone.
- `DEFAULT` untouched: no law changed; the only extraction-visible
  delta is the `batch_norm` price.

## Gates

- `pytest tests/test_derive_laws.py tests/test_cond_laws.py
  tests/test_discovery_dspec_synth.py` — **167 passed** (incl. new
  `TestTensorDerives` ×10 in derive_laws — tval/tapp dispatch, the
  `$leaf:` mint, declines, JSON round-trip, e-graph firing,
  deterministic term replay — and ×4 in dspec_synth — the tensor
  fit, the no-`key` honest refusal, leaf seeding, `_vkey` boxing).
- `pytest tests/test_law_serialize.py
  tests/test_lemma_certificates.py tests/test_egraph_internals.py
  tests/test_rules.py` — **141 passed** (the `apply_rule` subst
  path, cert replay and rule serialization are undisturbed).
- `ruff check` / `ruff format --check` on the touched shipped files —
  clean.  `ty check` — **0 errors**.  `lint-imports` — **4/4 kept**
  (core stays torch-free *and* numpy-free: the algebra is pushed,
  never imported).
