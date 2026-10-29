---
title: 'Retro: ADR-0007 layer 3 — atom laws from measured bindings'
type: retro
date: 2026-10-16
---

# Retro: ADR-0007 layer 3 — atom laws from measured bindings

Branch: feat/runtime-loop
Driver: `lab/mint/atom_law_loop.py` (new).  Artifact:
`project/retros/layer3-loop.json`.

Layer 1 minted definitional atoms (anti-unification); layer 2
minted handler bodies (`handler_synth`); this is the layer-3 half —
**an atom's law set synthesized from measured bindings**, run as a
closed loop over the three hand-built carrier atoms
(`om_decayed`, `applyd`, `decay_vec`).  The driver is one loop, not
three: per candidate schema it enumerates measured environments
(typed shape families x attr-domain products + the oracle's
`_synth_sites` window + real corpus matches), scores the tri-state
oracle outcome per env, mints guards (`auto_cond`, with a small
axis/attr bank fallback for the carrier-attr corner its generic
domain cannot reach) and attr derives (a solver-measured hook fitted
by `synth_dspec_multi`), then certifies through `verify_law` —
`derivable`/`subsumed-*` means the shipped library already implies
it, `primitive` means a genuinely new measured-true candidate.

## The applyd slot solve

`applyd(aff_diag(a,b), h) = a*h + b` is affine in `h`, so any LHS
affine in the state slot decomposes *per env* as
`L = (lhs(h+d) - lhs(h))/d`, `B = lhs(h) - L*h` (d drawn bounded
from 0).  Slot values are matched against the bounded expr bank and
intersected across envs — the value-vector dedup of `dspec_synth`
lifted to terms.  This replaced the blind `e1 x e2` pair product:
the true compose `(mul a1 a2, add (mul a1 b2) b1)` sat ~90k deep in
diagonal pair order — the quartic-hang class — while the slot solve
finds it in two passes of ~2.4k bank evals.  The same solve recovers
compose, add-merge, scale-absorb AND the matmul readout
(`matmul(E, applyd(f,h)) = applyd(aff_diag(E@a, E@b), h)` — h
broadcasts along the contracted axis so the Jacobian is diagonal in
it).

## Results (measured, all serial)

| atom | candidates | certified rows | rejected |
|---|---|---|---|
| applyd | 5 schemas + 18 solved | 18 (compose x8, addmerge x4, scale x4, readout, nobias) — scale[0] `subsumed-duplicate` via `xc_scale_applyd`, rest `primitive` | 1 (`addmerge_wrongstate`, deliberately false) |
| om_decayed | 6 | 4 — unfold/refold subsumed (refold hits 16 real corpus sites), `absorb_score` primitive, `rowcat` guarded `('axis','d1','D',-2)` primitive | 2 (`absorb_value`, `keycat_wrong`) |
| decay_vec | 4 | 3 — `merge` 96/96 equal, `merge_synth` (dspec *withheld*; solver+synth_dspec_multi minted `O=O1+O2, S=S1+S2`), `unit_mul` | 1 (`merge_sign_s1`, deliberately wrong) |

Distinct new measured-true laws modulo commutative spellings:
`affd.compose`, `affd.addmerge`, `affd.scale` (twin spellings),
`affd.readout`, `affd.nobias`, `omd.absorb_score`,
`omd.rowcat` (conditional), `dv.merge`, `dv.unit_mul`.

## The measurement caught a lowering bug

`decay_vec`'s executor read `sign = int(kw.get("sign", 1) or 1)` —
`or 1` silently rewrites `sign=0` to 1, so `dv(g,T,o,0)` evaluated as
`g^{k+o}` instead of the constant `g^o`.  The oracle flagged
`dv.unit_mul` `unequal` and the solver's (O,S) fits refused — the
loop declined to mint rather than shipping a wrong law.  Fixed
(`sign = int(kw["sign"]) if "sign" in kw else 1`); nothing shipped
mints `sign=0`, so the fix is behavior-neutral on existing laws
(256 carrier tests pass).

## Honest residuals

- `auto_cond`'s measured domain cannot reach attr-heavy carrier
  corners (concat dims, decay_vec attrs) — `omd.rowcat`'s guard came
  from the driver's typed-env axis bank, marked in the note.  The
  upstream fix is a domain override on `auto_cond_object`.
- Real-site coverage is thin for carrier-internal patterns (corpus
  terms are pre-rewrite; `applyd`/`om_decayed` arise during search).
  `real=0` rows lean on synthesized envs — typed by construction,
  but a narrower claim than a corpus-measured firing.
- `verify=primitive` says "not derivable by the shipped set", not
  "should ship" — cost/pay is unmeasured; admission is a separate
  gate (the mul_unsq_nat path).
