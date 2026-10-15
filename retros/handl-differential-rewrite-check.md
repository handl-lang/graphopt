# Retro — the differential rewrite check (catopt's engine vs handl's)

**Date**: 2026-10-15.  **Assets**: `lab/audit/handl_differential.py`,
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
form.  catopt currently emits `target_kind: None`, and the engines must
diverge.

Two ways out, and they are not equivalent:

| option | consequence |
|---|---|
| a rule form in handl (e.g. `Replace { path }`) | faithful; a handl-side design decision |
| a declared `catopt:alias` kind whose payload carries the path | loadable and lossless *via the `catopt:` record*, but handl's engine then sees an opaque created node — i.e. it **hides** the semantic difference |

The probe reports these laws **separately** rather than choosing for us.
Papering over a divergence inside a tool whose entire purpose is to find
divergences would defeat it.

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
2. **A regression guard**: `tests/test_handl_peer.py` pins 19 agree / 0
   disagree, so a change in *either* engine that breaks agreement fails a
   test.
3. **Two gaps, named**: the root-eliminating rule form (a real
   representational gap in handl's vocabulary) and the shape-sensitive
   guard limitation (a probe improvement).
