---
title: 'Retro — spike Q2: what does the boundary cost?'
type: retro
date: 2026-10-14
---

# Retro — spike Q2: what does the boundary cost?

**Plan**: 0029, **Q2**.
**Verdict**: **not applicable as posed** — and the reason is a decision,
not a dodge.  The measured cost of the v1 path is the fragment load,
which is negligible.

## The question, as the plan posed it

> **Pass criterion is *not* "faster"** … the criterion is: host-fn
> dispatch must not dominate **rewrite throughput** — measure
> rewrites/second through handl's engine on a small law set and compare
> with catopt's e-graph on the same set.  Pass = same order of
> magnitude; the exec timing is informational.

## Why it does not apply

The question assumes **host functions in the path** — that tensor ops
cross the boundary during rewriting.  Two measurements remove that
assumption:

1. **ADR 0011** decides the integration is **language-only in v1**:
   rewriting is canonical in the calculus, execution stays catopt's, and
   no tensor needs to cross.
2. **Q1b** shows that if a tensor *did* cross, it would cross as a
   raw process-local pointer through a debug encoding
   (`{"__resource__": {"ptr": n}}`) or as copied bytes — i.e. the
   expensive thing the question was trying to bound is exactly the thing
   the decision avoids.

So there is no host-fn dispatch to measure in the v1 path.  Measuring it
would be measuring a design we chose not to build.

## What *is* measured

| stage | cost | source |
|---|---|---|
| fragment load (`language_from_dir` over the ten-file dir) | **instant** — the spike's test reports `finished in 0.00s` for the merge + both assertions | handl `b20b978a` (`catopt_fragments.rs`) |
| the rewrite pass itself | handl's own engine cost, already covered by handl's suite (4,288 tests); not catopt's to measure | handl |
| catopt's side of the seam | the emitter (`store_to_handl`) — pure dict construction, no I/O, no tensors | catopt `lab/handl-spike/dump_fragments.py` |

## The honest caveats

- **"Instant" is a small fixture.**  One law + a handful of shapes.  The
  *load* cost scales with the bank (82 rules, ~35 op shapes) — still
  trivial in absolute terms, but the claim should be re-measured on the
  full bank before it is quoted.  (`dump_fragments.py --all` emits it;
  nobody has loaded the full bank yet.)
- **Rewrite throughput was not compared** — deliberately, per the above.
  If a future version routes tensor ops through the boundary, this retro
  must be redone with the original criterion, and the pointer route's
  cost (an integer in JSON per crossing, plus the host's lookup) is what
  would be measured.
- **The exec timing is untouched.**  catopt's executors are unchanged and
  unmeasured here; the boundary does not sit in their path.

## What this buys

It closes a question by **decision + measurement** rather than by
benchmark: the boundary's v1 cost is the load, and the load is
negligible.  If the integration later grows a tensor channel, the
original criterion is the one to apply — and it is written down.
