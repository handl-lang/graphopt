# Retro: handler-interpreted atoms — ADR 0007 layer 2 lands

Date: 2026-10-15
Context: ADR 0007 layer 2 ("handler-interpreted atoms — semantics
by denotation").  `genkernel` mints handlers whose kernel evaluates
the spec's own spelled body — the elementwise special case.  This
lands the generalisation the ADR asked for: a **structured handler
grammar** whose candidate bodies are proposed for a measured
subterm and verified numerically against the pre-mint term on the
measured domain — the same bar as a law.  Files:
`packages/mlopt-discovery/src/mlopt_discovery/handler_synth.py`
(the whole mechanism), `tests/test_discovery_handler_synth.py`
(37 tests, 100% module coverage).  Nothing else touched; core,
carriers and the sink are unchanged.

## The grammar (descriptors are plain data)

Five families, all evaluated by `eval_body` — no eval/exec, a
structured walk:

- `("ew", sub)` — elementwise replay; the `genkernel` tier
  re-spelled.  Proposed only for genuinely elementwise specs.
- `("contract", eq, lhs, rhs)` — `einsum(eq, EW(lhs), EW(rhs))`;
  the two factors are elementwise trees over metavar subsets that
  cover the spec's metas.  `eq` ∈ 7 closed equations (the four
  batched-matmul orientations, transposed output, matvec, inner
  product).
- `("fold", a, b, state, axis)` — `acc_t = ew(a)_t ⊙ acc_{t-1} +
  ew(b)_t` walked along `axis`, operands broadcast into the step
  space (the scan tier's `_seq_args` convention).
- `("fold_emit", a, b, state, emit, axis)` — the recurrence plus a
  per-step projection `out_t = ew(emit)` where `emit` may read the
  running accumulator through the reserved `"$acc"` leaf; outputs
  stack back on `axis`.
- `("fold_zip", steps, state)` — the flat chronological fold:
  `acc = a_i ⊙ acc + b_i` over a tuple of metavar pairs — the
  semantics an unrolled `applyd` nest already spells.

Elementwise factors come from a bounded expression grammar
(`_ew_exprs`): 7 unary + 4 binary ops, ≤ 2 op nodes, spec
constants admitted as atoms.  Proposal enumeration is
deterministic, simplest-first (total op count), round-robin across
columns (partitions / states / pairings) so no column's deep
region starves another's cheap candidates.

## Verify-or-refuse, measured

`verify_body` draws fp64 operands per domain entry (paired draws —
same seed for every candidate) plus an adversarial re-draw under
an offset seed; the gate is `allclose(rtol=1e-9, atol=1e-12)`
against the spelled spec evaluated through the ambient bindings.
`mint_handlers` returns `(handlers, report)` — every item gets a
verdict row: minted, covered, or refused (with the measured
best-seen error and which of three refusals applied: outside the
grammar, grammar exhausted, budget cap).

The measured table (seed 0, fp64, one domain binding per spec):

| site spec | verdict | family | tried | max_err |
|---|---|---|---|---|
| `matmul(mul(S,D),V)` — the `om_decayed` element | minted | contract | 13 | 8.9e-16 |
| `matmul(mul(mul(S,D),M),V)` — masked inner-chunk | minted | contract | 31 | 1.8e-15 |
| same, batched `(B,T,K)` operands | minted | contract | 13 | 0.0 |
| closed-form `H·∏a + Σ b·(∏a/∏≤a)` | minted | fold | 10 | 4.4e-16 |
| `Q ⊙ (running acc)` sequence | minted | fold_emit | 417 | 1.3e-15 |
| `applyd(aff_diag(A,B), applyd(aff_diag(C,D),H))` | minted | fold_zip | 292 | 0.0 |
| `sum(X, dim=-1)` | refused | — | 0 | — |
| `exp(matmul(X,Y))` | refused | — | 1200 | best 1.82 |

The pay the task named is real: `om_decayed`'s `(s⊙d)@v` — a
pattern that today needs a hand-written executor row
(`om.TORCH_BINDINGS["om_decayed"]`) — mints as
`("contract", "...ik,...kj->...ij", ("mul","D","S"), "V")`,
verified fp64 against the spelled term, and the minted op lowers
through `minted_sink` end to end.  Same for the masked
inner-chunk form `((s·d)·m)@v`, the chunked fold's closed form
(`fold`), a `q ⊙ acc` projection sequence (`fold_emit`), and the
two-step unrolled `applyd` nest (`fold_zip` — bitwise 0.0).  The
one-step `fold_zip` also mints the carrier's base step `a⊙h+b`
verbatim.

## The falsification read — ADR 0007 #3 ("handler grammar too weak")

**Not falsified.**  Every one of the three non-elementwise
template shapes the ADR names — contraction, fold, fold+project —
produced a verified handler against its pre-mint subterm inside a
≤ ~420-candidate bounded search.  The grammar is strong enough to
write a scan: `("fold", "A", "B", "H", -2)` IS the sequential
recurrence, verified against the spelled closed form.

Honest boundaries, measured not guessed:

- **The seq-fold needs a spelled fold.**  `("fold",…)` cannot
  verify against `applyd(aff_diag(A,B),H)` — the ambient `applyd`
  is one map application `a⊙h+b`, a different function of the same
  bound values.  Verification only exists where the pre-mint term
  itself spells the fold (the closed form, or the unrolled nest
  `fold_zip` covers).  A site whose intended reading is the fold
  of sequence operands is a Layer-3 question — the atom names a
  semantics nothing spelled — not a Layer-2 verification failure.
- **Reductions/softmax are outside the grammar** — `tried: 0`
  refusals.  Named template shapes missing today: reduction axes
  (`sum`/`mean`/`prod` as first-class templates), row-normalised
  contractions (softmax's `exp/sum` shape), and contractions over
  more than two factors.
- **Emit-only operands required a coverage fix** — the first
  enumeration forced every non-state meta into `a`/`b`, so
  `q ⊙ acc` sites (where `q` feeds only the projection) were
  unreachable.  The shipped enumeration covers leftovers by the
  emit; a meta may serve both roles.
- **Budget is the honest cap** — `exp(matmul)` burned all 1200
  contract candidates (best err 1.82).  Deep folds in
  `fold_emit`-shaped space cost ~400 tries; anything deeper is a
  budgeted gap, not an unseen one.
- **Spec constants only.**  The atom pool is the spec's own
  constant leaves plus metas; fresh constants (e.g. a `0.5` a
  handler would need) are unproposable — a named gap.
- **Emit metas can't mix freely.**  An emit expression draws from
  rest ∪ `{$acc}` ∪ consts; deeper emit trees (two ops inside the
  projection) are bounded out.

## Delivery + provenance

A minted entry is a handler-table row `{pattern, kernel, args,
handler_body, provenance}` — the same shape `genkernel`'s tiers
mint, so `claim`/`handle` binds it and `handlers_fragment` wraps
it as cells with no new machinery.  `minted_bindings` interprets
the descriptor (verified semantics, not a replay); `minted_sink`
extends `OpTable.core()` exactly like `gen_sink`.  `handler_record`
/`handler_from_record` JSON-round-trips the atom — grammar path,
verified domain, seed, tried count, max_err — the minted atom is
provenance data, not live code.
