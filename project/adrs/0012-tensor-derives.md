# ADR 0012 — tensor-valued derives: the `$leaf:` channel and the pushed tensor algebra

**Status**: proposed — **implemented** (this commit lands the seam;
the conv+BN fold law itself remains un-authored, per the weight-space
lane's no-hardcoding rule).
**Date**: 2026-10-18.
**Depends on**: ADR 0004 (abstraction as move — the `dspec` DSL this
extends), ADR 0007 (semantic-atom minting), the derive-as-data
migration recorded in `project/retros/derive-declarative.md` /
`derive-to-data.md` / `derive-root-ops.md`.
**Grounded in**: `project/retros/weightspace-fold.md` (the measured,
gate-by-gate refusal) and `project/retros/derive-root-ops.md` (the
residual after `sqrt`/`rsqrt` landed: *the derive value domain is
`int | float`* — gate 4).

## Context

The weight-space probe measured the conv+BN fold

    W'[o] = W[o]·γ[o]/√(σ²[o] + ε)      b' = (b − μ)·γ/√(σ² + ε) + β

as **structurally derivable**: `dspec_synth` fits `W' = W·scale` and
`b' = (b−μ)·scale + β` once a `scale` leaf exists, and after
`18de018` the DSL spells `scale = γ·rsqrt(σ²+ε)` over scalars.  What
refuses — measured, twice — is the **value domain**:

- a derive emits `{"$attr:NAME": value}` feeding RHS *attrs*, which
  must round-trip through `attr_to_data` (JSON-safe scalars/tuples —
  an array raises `TypeError: unserializable attribute value`) and
  through `_attr_key` (repr-keyed — an array's repr is its payload);
- `Const.value` is typed `int | float` — a tensor cannot be a `Const`;
- `dspec_synth`'s fingerprinting hashes sample values — an
  unhashable array raises `TypeError: unhashable type:
  'numpy.ndarray'` (probe gate 4);
- `bound` maps metavars to *terms* (`Param` = name + shape), not
  tensor values — the array behind a leaf lives in the adapter's
  `source_tensors`, a dict core never sees.

Plus the two *content* gaps the same retro named: no law LHS names
`batch_norm` (unchanged — the law is derivable content, not
capability, and this ADR does not hand-write it) and `batch_norm` is
absent from `_OP_FLOPS` (priced by this change — see *Decision*).

## Decision

### 1. The derived value is a *term*, and it lands under `$leaf:`

A `dspec` key may carry the **`$leaf:`** prefix alongside the
existing bare/`$attr:` keys:

    {"$leaf:Wn": ("mul", ("tval", "w"), <scale expr>)}

Under `$leaf:NAME` the expr's value is packaged as a **leaf term**
and emitted under the bare metavar key — `{"Wn": Param(...)}` — so
the RHS pattern's `"Wn"` metavar resolves to it at instantiate time,
exactly like an LHS-matched binding.  Scalar results mint `Const(v)`;
array results mint `Param(name, TensorType(shape))` through the mint
seam (below).  This is the one contract change: `derive` output keys
are no longer all `$attr:` — non-`$attr:` keys bind RHS *term*
metavars.  `apply_rule` converts such values to e-ids via
`add_term` (subst stays metavar → eid; `ProofEdge.subst` and cert
serialization are unchanged); `instantiate_pattern` and
`_term_instantiate` already drop `subst[name]` in place — the
term-level sites need no change.

### 2. Tensor values are *data handled by a pushed algebra*, never objects core inspects

`mlopt-core` stays torch-free **and** numpy-free (the import-linter
`forbidden` contract).  The derive DSL therefore treats a tensor
value as **opaque** — it is created, transformed, and named only by
an adapter-pushed **`TensorAlgebra`**, registered the same way the
torch binding tables and the carrier value-shape resolver are
pushed (core is a sink, never a puller):

    register_tensor_algebra(source=…, mint=…, ops=…, key=…)

- `source: term -> array | None` — resolve a bound leaf's value
  (the adapter closes over `source_tensors`; a `Param` resolves by
  name).  Feeds the new **`("tval", T)`** expr: the tensor twin of
  `("const", T)` / `("vshape", T)`.
- `ops: {name: fn}` — elementwise implementations the grammar
  dispatches into: a `sqrt`/`rsqrt`/`exp`/`reshape`-style table.
  Consumed two ways: (a) **`("tapp", name, e1, …)`** — the generic
  apply, so the vocabulary boundary is the *table*, not the grammar
  (the next measured op — `tmean`, `texp`, a slice — needs no DSL
  change); (b) as a **fallback in the `_DUNOPS`/`_DBINOPS`
  handlers**: scalar operands keep today's exact semantics; a
  non-scalar operand dispatches to `ops[op]` when registered, else
  to the Python operator (which backends already overload —
  `a * b`, `a + b`, `-a`, `1.0 / a` all work on arrays).  The only
  ops that *need* the table are the `math.*` ones — `sqrt`,
  `rsqrt` — plus `tapp`-only names like `reshape`.
- `mint: (array, hint) -> (name, shape) | None` — register a derived
  array (into `source_tensors`) and return the leaf's identity.
  **`name` must be a deterministic function of the array's content
  and the hint** — certificate replay re-runs `derive` on terms, so
  a counter-style fresh name would mint a different `Param` on
  replay and break the recorded derivation (the `factored.py`
  `_fresh_name` convention is *not* adequate here; a content hash —
  `f"{hint}__{sha256(bytes)[:16]}"` — is).  `shape` comes back as
  data (`tuple[int, ...]`) so core builds the `TensorType` without
  touching the array.
- `key: array -> hashable` — a content fingerprint for
  `dspec_synth`'s value signatures (e.g.
  `(shape, dtype, bytes-hash)`).  Without it tensor samples dedupe
  by identity and every tensor fit honestly refuses — no crash, no
  false mint.

Unregistered capability declines, never fabricates: `tval` with no
`source`, `tapp` with an unregistered name, `$leaf:` with no `mint`
all veto the whole spec — the strictness contract unchanged.

### 3. The domain is *per-leaf opaque tensors*, not a typed buffer language

The derived value is one array per `$leaf:` key, with `mint`
deciding its name and declared shape.  We deliberately did **not**
add: named-buffer environments, multi-output derives, `None`-dim
payload shapes, or a tensor-typed *pattern* language.  The measured
refusal needs exactly "compute one weight tensor from bound weight
tensors + scalar attrs" — the fold's `W'`/`b'` and `batch_norm`'s
running stats are all that shape.  Anything richer is speculative
widening the retros do not justify.

### 4. `batch_norm` joins `_OP_FLOPS`

Priced `4` — the eval-mode spelled work `(x−μ)·scale+β`: sub + mul +
mul + add ≈ four elementwise passes over `n_out`, the same "priced
like the subgraph it replaces" convention `decay`/`om_decayed` use.
`group_norm`/`instance_norm` stay unpriced — unmeasured for this
gap.

## Why this seam and not the alternatives

| alternative | rejected because |
|---|---|
| widen `Const.value` to arrays | `Const` is repr-keyed leaf identity; an array's repr is its payload — interning, hashing and `_attr_key` all break, and `Const` *means* scalar. |
| tensors as JSON-safe attrs (nested lists) | quadratic payload in every attr key, dtype/device lost, and attrs are metadata — the `attr_to_data` contract is right to refuse them. |
| derive emits an *unevaluated recipe*, instantiated lazily | pushes evaluation into every instantiate site (e-graph, term-level, cert replay, meta composites) — four seams instead of one; the recipe stays expressible anyway (the `dspec` *is* the recipe). |
| a new IR leaf (`DerivedParam` holding the array) | ripples through every `isinstance` ladder — match, serialize, cost, lowering.  `Param` + `source_tensors` registration is the existing seam (`factored.py`'s low-rank pass already registers derived weights that way); a folded weight *is* a `Param`. |

## Consequences

- **Unblocks (measured)**: `weightspace-fold.md` gate 4 — the last
  capability refusal; and gate 6 (pricing).  The fold's remaining
  gaps are *content*: a `batch_norm` LHS law (derivable now that the
  value domain admits it) and synth enumeration depth ≥4 (a budget
  question, not a domain one).
- `dspec_synth` fingerprints values through a hashable `_vkey` box —
  the `TypeError: unhashable` crash is gone; tensor leaves
  (`("tval", T)`) join the leaf bank when a `source` is registered.
- **Known limits, stated honestly**: (a) ~~`meta.synthesize_rules`
  still requires RHS *term* metavars to be LHS-bound — a `$leaf:`
  rule cannot be auto-composed (the `@i:` placeholder machinery is
  attr-only); term-level namespacing is future work, refused rather
  than guessed today.~~ **Closed (2026-10-18)**: term-level
  placeholders landed — a `$leaf:` output travels as an `@i:` leaf
  metavar refilled by the composite `derive` at instantiate time
  (`retros/leaf-compose-seam.md`).  (b) The mint must be
  deterministic — a non-deterministic adapter breaks cert replay; the
  contract is documented, not enforced.  (c) A `$leaf:` value that is
  neither scalar nor mintable declines — there is no third landing
  zone.
- `DEFAULT` is untouched: no law changes, so extraction behavior for
  existing callers is unchanged except the `batch_norm` price.
  (Follow-up: `BATCH_NORM_RULES` shipped as an *opt-in* list —
  `DEFAULT` stays untouched pending admission.)
