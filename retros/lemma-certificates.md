---
title: Lemma certificates — `derivation` upgraded from provenance to proof
type: retro
date: 2026-10-14
---

# Lemma certificates — `derivation` upgraded from provenance to proof

The axiom/lemma split (`axiom-lemma-split.md`) recorded each non-kernel
rule's `derivation`: the names of shipped rules forming ONE measured
derivation of the rule's instance.  That field was *metadata* — a claim
that a derivation exists, verified once by `--emit-basis` and never
again.  This change materializes it as a **replayable certificate**:
saturate the lemma's instance under the recorded premises alone,
reconstruct the positional derivation through the shipped proof
machinery, and verify it on real terms, strictly.

Reproduce:

    .venv/bin/python tools/law_lemma_cert.py            # ~5 s, CPU
    .venv/bin/python tools/law_lemma_cert.py --demo silu_mul_form
    .venv/bin/python tools/law_lemma_cert.py --with-layout \
        --probe right_factor_linear
    .venv/bin/python -m pytest tests/test_lemma_certificates.py -q

## 1. The mechanism — reuse, not new machinery

The answer to "can the saturation engine's witness machinery capture
law-level derivations" is **yes, verbatim**.  `verify_law` already
interned both sides of a law instance and asked
`EGraph.certificate`/`verify_certificate` whether the merge replays;
the tool simply keeps the `Certificate` object instead of collapsing
it to a bool.  The verdict ladder:

* **linear** — `certificate()` replays standalone (strict, no trusted
  stubs): a genuine step list `[(rule, path, lhs, rhs, bindings)]`.
* **enumerated** — `certificate()` emits only e-graph-dependent stubs,
  but `all_proofs` (bounded BFS over the positional rewrite space)
  surfaces a linear path, wrapped into a `Certificate` and verified.
* **saturation-only** — merged, no linear derivation within fuel.
* **gap** — the premises fail to merge at all: annotation drift.

New surface, all small: `cert_to_data` / `cert_from_data` — the
certificate↔JSON codec — in `catopt_core/egraph/certs.py` (re-exported
by `catopt_core.egraph`), the twin of `laws.serialize.law_to_data`:
terms through `term_to_data`, bindings tagged `{"term": …}` /
`{"attr": …}` (metavar vs `"$attr:"` keys), rules referenced *by name*
— hooks are code, data cannot carry them, same honest posture as
`missing_hooks`.  `tools/law_lemma_cert.py` runs the measurement;
`tests/test_lemma_certificates.py` pins it (7 tests).

## 2. The measurement — all 15 materialize, all one step

15 rules carry `derivation` (13 `lemma` + 2 `redundant`).  Under
premise-only saturation, **all 15 get linear certificates — every one
exactly 1 step, exactly the recorded premise** (`premise_cover`: the
replayed step's rule is the annotation's name, no extras):

| direction | count | who |
|---|---|---|
| `rhs->lhs` | 13 | every inverse twin + `weight_factor_matmul` |
| `lhs->rhs` | 2 | `weight_distribute_matmul`, `silu_mul_form` |

The direction column is the finding, not a formality:

* **Inverse twins prove `rhs->lhs`.**  `silu_fold`'s certificate is
  `silu(x) → mul(x, σ(x))` — literally the *axiom's* forward step
  applied to the lemma's RHS instance.  The lemma's own direction
  (`mul(x,σx) → silu(x)`) emits an e-graph-dependent stub: it is
  *unreachable by forward premise rewriting* — `silu_expand` can
  never produce a `silu` node.  The certificate makes precise what
  "derivation" means for a twin: the equality is witnessed by the
  axiom, read symmetrically; the *rewrite* exists only because the
  lemma ships.  This is `silu_fold`'s mediation case spelled out —
  its witness is `silu_expand`, full stop.
* **`weight_distribute_matmul` proves `lhs->rhs`** — it is the
  same-direction alpha-duplicate of its premise
  `distribute_matmul_over_add`, so the cert is the axiom's own step on
  an alpha-equivalent instance.  (Its classmate
  `weight_factor_matmul` is the factor-direction duplicate, hence
  `rhs->lhs` like the twins.)
* **`silu_mul_form` proves `lhs->rhs` at path `(0,)`** — the only
  composite derivation: `silu_expand` fires *inside* the `mul`
  context, not at the root.  The cert step records the non-root path
  — the one lemma whose proof is more than "the axiom at the root".

Nothing is saturation-only and nothing gaps: the annotation table is
not just premise names that happen to be sufficient, it is a set of
concrete one-step proofs — and the tool now *replays* each of them on
every run.  A lemma whose recorded proof stops replaying after a
library edit now fails as `gap`/`saturation-only` instead of rotting
as a stale comment — the drift check the axiom/lemma retro asked for.

## 3. The honest boundary — "unreplayable" refines to "reconstruction-hard"

`--probe NAME` runs the same ladder on any rule against
`universe − {name}`.  The measured boundary cases:

| probe | universe | verdict | detail |
|---|---|---|---|
| `right_factor_linear` | `ALL_RULES` | **gap** | primitive — consistent with its axiom mark |
| `right_factor_linear` | `+LAYOUT` | **enumerated** | `certificate()` stubs both directions; `all_proofs` finds 8 linear derivations, shortest 4 steps |
| `linear_to_matmul_t` | `+LAYOUT` | **linear** | `rhs->lhs`, 1 step via `linear_from_matmul_t` (de-facto twin) |
| `matmul_transpose_rev_bb` | `+LAYOUT` | **linear** | 3 steps: `linear_from_matmul_t_bare → linear_to_matmul_t → matmul_transpose_rev_bd` |
| `comm_add` | `ALL_RULES` | **gap** | soundness control: primitives don't self-derive |

`right_factor_linear` was the depth-2 retro's "lone
derivable-but-unreplayable law".  The probe refines that verdict: the
merge is real AND a linear derivation exists — `linear_to_matmul_t`
at `(0,)`, `linear_to_matmul_t` at `(1,)`, `right_factor_matmul` at
`()`, `linear_from_matmul_t` at `()` — the connect machinery just
can't *reconstruct* it.  The failure is structural: neither endpoint's
root enode is a rule-application RHS (so target-expansion finds no
app), the heads differ (`add` vs `linear`, so congruence fails), and
`_edge_path`'s fallback searches **root-level rewrites only** — the
derivation needs fires at child positions `(0,)`/`(1,)`.  `all_proofs`
enumerates the positional space and finds it.

Two consequences worth recording:

* **"Saturation-only" is a bound on reconstruction, not on
  existence.**  Under pure rule saturation every merge has some
  linear witness in principle — each union IS a rule firing, and
  connect+congruence can compose them.  The genuinely non-linear
  class is non-local-pass merges (`pair_shared_input_*`, manual
  `union` without `witness=`), which never occur in law verification.
  The residual honest boundary is *budget*: a derivation longer than
  `all_proofs`' fuel/steps cap would still read saturation-only.
* **The enumerated proof is 2nd-generation**: it routes through
  `right_factor_matmul`, itself a *lemma*.  A law derivable only
  through another derivable law is exactly the stratification the
  depth-2 retro measured (`matmul_transpose_rev_*` rank-2 chains) —
  now visible as concrete step sequences, not just premise sets.

## 4. Serialization — the store loop closes

`cert_to_data` makes a certificate the same kind of record a law
already is.  `silu_mul_form`'s cert → JSON → `cert_from_data` →
`verify_certificate(strict=True)` replays to
`(mul (mul g, (sigmoid g)), u)` — demonstrated by `--demo`.  The
record carries `src`/`dst`/`steps[]`, each step
`{rule, path, lhs, rhs, bindings, egraph_dependent, note}`; the
lemma-store row is now complete: **pattern + cond + derivation +
replayable proof**, where `derivation` is the premise index and the
certificate is the checkable artifact.  `--json` emits all 15
certificates as data.

## 5. Honest limits

* Everything inherits the instance-level, budgeted posture of the
  coherence probes: "linear" means one certificate replayed on *the
  law's registered instance*, not a proof of the law in general —
  though a step sequence that replays on one instance replays on any
  instance of the same binding shape.
* `enumerated` certificates depend on `all_proofs` fuel; a deeper
  derivation could exist beyond `max_steps=10`/`fuel=200k` and still
  read `saturation-only`.
* Direction is a convention the data makes explicit: a `rhs->lhs`
  cert proves the *equality*, not the lemma's rewrite direction —
  symmetric reading is sound but the record keeps which way the
  axiom actually fired.
* The codec references rules by name; a record decoded against a
  rule set missing a name can't verify (raises "unknown rule") —
  hooks, as always, are not data.
