---
title: 'Retro — the `postnorm` family: `log_softmax`''s wrapped-reduction gap closed'
type: retro
date: 2026-10-17
---

# Retro — the `postnorm` family: `log_softmax`'s wrapped-reduction gap closed

**Scope**:
`mlopt_discovery/handler_synth.py`, `lab/audit/grammar_probe.py`,
`tests/test_discovery_handler_synth.py`.

**Depends on**: `project/retros/grammar-gaps.md` (the residue this
closes), ADR 0007 layer 2 (the handler grammar), plan 0025 §B
(grammar extension by measured refusal, not taste).

## The residue

`grammar-gaps.md` named it: *"`log_softmax` (a real invisible corpus
op, 2 sites) — a normalization whose reduction result is wrapped in a
unary (`log . sum . exp`) or a `logsumexp` reduction; the `norm`
template covers a bare reduction operand only, and `logsumexp` has no
torch binding."*  A `norm` body computes `op(lhs, red(rhs, axis,
keepdim))` — the reduction output is a bare operand of the outer op.
`log_softmax(x) = x - log(sum(exp x))` needs the reduction *result*
wrapped in `log`, so no `norm` body verifies — a grammar-coverage
refusal (candidates proposed, none verify).

## The extension

`("postnorm", op, lhs, post, red_op, rhs, axis, keepdim)` —
`op(lhs, post(red_op(rhs, axis, keepdim)))`, where `post` is a unary
from the new `POST_UNARY` vocabulary.  Deliberately minimal:

- `POST_UNARY = ("log",)` — only the measured wrap.  Other wraps are a
  grammar extension, not a candidate.
- The family is **seed-only**: `_canonical_postnorm_seeds` proposes
  `op(meta, post(red_op(ew, axis, keepdim)))` — a bare-meta lhs (the
  value being centered/scaled) over the shallow elementwise rhs
  vocabulary, `op`/`post`/`red_op`/`axis` over their banks.  ~160
  candidates for a 1-meta spec.  No full column walk exists — a
  non-canonical `postnorm` body is a declared grammar gap, not hidden
  search.
- Same shape gate as `norm` (proposed only when the spec's top op is
  in `NORM_OPS`); `_admitted_families` admits it for any non-empty
  meta set; `_GRAMMAR_VERSION` bumps to `handler-grammar-v2` (the
  vocabulary changed, so the provenance stamp must too — one existing
  pin updated).
- Records round-trip unchanged: `body_to_data`/`handler_from_record`
  are generic over the descriptor tuple, and `provenance.family`
  reads `body[0]` = `"postnorm"`.

## Measured

- `log-softmax` spec (`sub(X, log(sum(exp X, -1, keepdim)))`) —
  **minted** `("postnorm", "sub", "X", "log", "sum", ("exp","X"),
  -1, True)`; verified at the fp64 gate, and the descriptor agrees
  with `torch.nn.functional.log_softmax` to 1e-12 (pinned by test).
- `grammar_probe.py` battery gains the `"log-softmax"` spec so the
  census reports the family's reach.
- The other grammar-gap residue — the *stabilized* (clamped) scan —
  stays open: it is a non-affine two-state recurrence, a semantics
  extension, not a vocabulary one (unchanged verdict).

## Gates

- `pytest tests/test_discovery_handler_synth.py` — **47 passed**
  (4 new: `eval_body` on the descriptor, the `log_softmax` mint, the
  shape gate, the record round-trip).
- `ruff check` / `ruff format --check` on the touched files — clean.
- `ty check` — 0 errors.
