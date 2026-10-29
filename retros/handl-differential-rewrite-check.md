---
title: Retro — the differential rewrite check (catopt's engine vs handl's)
type: retro
date: 2026-10-15
---

# Retro — the differential rewrite check (catopt's engine vs handl's)

**Assets**: `lab/audit/handl_differential.py`,
`tests/test_handl_peer.py`.  **Enabled by**: the peer (plan 0030 step 2),
which is what lets catopt drive a *second* rewrite engine at all.

## What it does

For every law in the shipped bank it synthesizes one term matching the
left-hand side, rewrites it with **catopt's** engine and with **handl's**
(through the peer CLI), renders both results as `{kind, children}`, and
compares them.

Two independent engines applying the same law to the same term is
evidence one engine cannot give — the cheapest form of "certified
rewrite" the project has.  It is also the first thing that uses the
integration for something other than proving the integration works.

## Result — whole bank, all 83 laws accounted for

| bucket | count |
|---|---|
| compared | 19 |
| **agree** | **19** |
| **DISAGREE** | **0** |
| skipped (leaf rhs) | 3 |
| root-eliminating | 5 |
| declined by an engine | 37 |
| guard raised (synthesized term) | 19 |

**Where both engines fire, they agree.**  The law bank survives an
independent implementation — that is the headline, and it is a stronger
statement than any single-engine test can make.

The 64 laws that could not be compared are each reported **with a
reason**, and two of those reasons are findings.

## Finding 1 — root-eliminating laws cannot be expressed (5 laws)

`double_neg` is `neg(neg(a)) → a`: the law **drops its root** and returns
a subterm.  handl's `TransformRule::Declarative` always *creates* a node
— its `target_kind` is a `String` — so such a law has no faithful rule
form.  catopt used to emit `target_kind: None`, and the engines diverge.

Two ways out, and they are not equivalent:

| option | consequence |
|---|---|
| a rule form in handl (e.g. `Replace { path }`) | faithful; a handl-side design decision |
| a declared `catopt:alias` kind whose payload carries the path | loadable and lossless *via the `catopt:` record*, but handl's engine then sees an opaque created node — i.e. it **hides** the semantic difference |

The probe reports these laws **separately** rather than choosing for us.
Papering over a divergence inside a tool whose entire purpose is to find
divergences would defeat it.

### Addendum (2026-10-16) — what `target_kind: None` actually did, and the fix

The gap was investigated.  Two results.

**1. `target_kind: None` was silently wrong — a bug in catopt's bridge.**
`TransformRule::Declarative.target_kind` is a Rust `String` (not an
`Option`), but `language_from_dir` parses `transforms.yaml` with
`serde_yaml::from_str::<Vec<TransformDef>>`, and serde_yaml coerces the
YAML `null` scalar into the **string `"null"`**.  So the row loaded, and
handl's engine (`walk.rs::fire_untraced` → `step.rs::build_rewrite`,
`kind: target_kind.to_string()`) **fabricated a node of kind `"null"`**.
Firing the old `double_neg` row on `neg(neg(a))` gave handl
`{'kind': 'null', 'children': [a]}` where catopt gives `a` — a *silent
lie*, not a refusal (and handl's `validate_transforms` only *Warns*
about the unknown kind, so assembly still succeeds).

**Fixed** (catopt `8548243`): the bridge now **refuses explicitly** — for
a bare-metavar rhs it emits `rule: !Native "catopt:eliminate_root"`
naming the missing operation.  handl's engine then rejects it loudly
(`"requires native rewrite 'catopt:eliminate_root' but no native
registry is wired"`), the bank still loads (83 transforms), and the law
still round-trips through `catopt.law`.  The probe is unchanged — it
classifies root-eliminating laws before firing — so the divergence stays
visible, and is now *louder* rather than hidden.

**2. No faithful form exists in handl's *existing* vocabulary.**  Checked
against the engine: `Splice` lifts a child's children but the
`Declarative` node still exists; `KeepPath` is a *child step*, never the
rewritten node; `Declarative` always creates, so a node cannot be
replaced *by* a child.  `TransformRule::Native` is the only mechanism
that *can* return a subterm (`ctx.remap_child(i)` returns an existing
`AstRef`), but the peer wires no registry (so it errors) and the
implementation would be catopt's Rust — handl's engine would be running
catopt's code, making the comparison **circular** for exactly these 5
laws.  So `Native` is a *refusal marker*, not a fix; the honest fix is
the handl-side `TransformRule::Replace { path }` proposal recorded in
plan 0030 step 3a (a handl design decision — handl's `transform_def.rs`
was deliberately not edited).

Differential numbers are **unchanged** by all of this: 19 agree / 0
disagree / 5 root-eliminating / 3 leaf-rhs skipped / 37 declined / 19
guard-raised.

**Related, unfixed (no bank law hits it):** the bridge emits
`node_kind: None` for a *bare-metavar lhs* (`TransformDef.node_kind` is
also a `String`), which serde_yaml likewise coerces to the string
`"null"`.  No shipped law has a bare-metavar lhs (checked: 0), so it is
latent; it is pinned only by the synthetic
`test_bare_metavar_root_refuses_as_native`.  A
`node_kind` has no `!Native` analogue (it is the dispatch key), so this
one has no clean refusal either — noted here rather than fixed.

### Addendum 2 (2026-10-16) — the rule form landed; the 5 laws are compared

The `Replace { path }` proposal was taken by handl: **handl ADR-0127**
(`crates/handl-language/src/fragments/transform_def.rs` for the variant,
one arm in `plugin-engine-transform-rewrite/src/walk.rs::fire_untraced`,
validation in `plugin-engine-transform/src/validate.rs`).  An unresolvable
path is a hard error, never a fabricated node — the failure mode Finding 1
records is now impossible by construction.

catopt's bridge emits `!Replace { path }` for a bare-metavar rhs bound in
the lhs (the metavar's lhs path), so the whole bank maps faithfully; the
`!Native` refusal survives only for a bare-metavar rhs with no bound path
(no shipped law).

**The probe changed in two honest ways** (it hides nothing — `DISAGREE`
stays 0):

1. the obsolete `_eliminates_root` pre-classification is gone; the laws
   are fired and compared like any other structural law;
2. the "structural laws only" filter now filters the **rhs** only — its
   documented intent.  Filtering the lhs too had needlessly skipped four
   laws whose *lhs* carries a leaf but whose rhs is structural
   (`id_add`, `id_mul`, `softsign_fold`, `pow_to_square`); a leaf in the
   synthesized **input** is perfectly comparable.

**Re-measured — whole bank, all 83 laws accounted for:**

| bucket | before | after |
|---|---|---|
| compared | 19 | **24** |
| **agree** | **19** | **24** |
| **DISAGREE** | **0** | **0** |
| skipped (leaf rhs) | 3 | 1 |
| declined by an engine | 37 | 39 |
| guard raised (synthesized term) | 19 | 19 |

`root-eliminating` is no longer a bucket (nothing is skipped for it); the
probe now prints the five per law:

```
  root-eliminating laws (bare-metavar rhs — handl !Replace, ADR-0127):
    - id_add: agree
    - id_mul: agree
    - double_neg: agree
    - transpose_noop: declined
    - chunk_single: declined
```

So **three of the five agree inside the probe**; `transpose_noop` and
`chunk_single` are **declined by catopt's own engine** on the probe's
synthesized term — their guards (`axes-noop`, `attr-eq CK 1`) need
concrete attrs the generic synthesis cannot supply (that is Finding 2, not
a divergence).  Given a guard-satisfying term **both engines return the
subterm**, so **all five agree wherever both fire** — pinned by
`tests/test_handl_peer.py::test_root_eliminating_laws_agree_on_satisfying_terms`.

The 19→24 gain is the five root-eliminating laws (minus the two the guards
decline) plus the four the lhs-filter had hidden.

## Finding 2 — one synthesized term per law is not universally sound

19 laws' guards **raise** on a synthesized term (bare `f64` variables):
they are *shape-sensitive* — they consult the shape of a child, and
`term.typ.shape` on a bare variable is not there.  Counted separately so
it cannot mask a real disagreement.

That is a limitation of the *probe*, not of either engine, and it points
at the obvious improvement: synthesize terms the guards accept (or run
the comparison on real corpus terms, where shapes are known).

## What this does **not** prove

- **Exhaustive equivalence.**  One term per law is a smoke-level check;
  it looks for disagreement, and one would be enough to matter.
- **That the laws are *correct*.**  This compares two engines'
  *rewriting*, not their semantics — the numeric oracle and the
  certificate replay are what say a law is true.  A law can be faithfully
  rewritten by both engines and still be wrong.
- **Anything about execution.**  No tensors cross (ADR 0011).

## Value

1. **A new asset**: cross-engine agreement over the bank, measured, with
   the non-comparable laws named.
2. **A regression guard**: `tests/test_handl_peer.py` pins **24 agree / 0
   disagree** (was 19 — Addendum 2), so a change in *either* engine that
   breaks agreement fails a test.
3. **One gap closed, one named**: the root-eliminating rule form was a real
   representational gap in handl's vocabulary and is now **closed** (handl
   ADR-0127); the shape-sensitive guard limitation (a probe improvement)
   remains and is what still keeps `transpose_noop`/`chunk_single` out of
   the compared set.
