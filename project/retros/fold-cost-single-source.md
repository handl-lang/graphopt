# Retro — `FOLD_OPS`: one compile-time fold alphabet; the premise-pay leg flips

**Date**: 2026-10-19.  **Follows**: `enablement-floor.md` — its named
open item: *"a cost model that credits compile-time weight folds would
flip the measured pay leg — a cost-model gap, not a law-truth gap."*
**Commits**: `ccfb9ba` (the change + tests), this commit (retro +
GOALS).

## The gap was drift, not a missing model

The machinery already existed: `dag_cost`'s per-node walk, the
e-class-level param-only discount in `extract_best`, and
`_generic_overhead` all consult `cost._folds_to_param` — "a subtree
closed over Param/Const leaves is compile-time work, priced ~0
marginal", which is exactly the generic principle required (a
structural property, never `op == "conv2d"`).  What had drifted was
the *alphabet* the predicate consulted:

| consumer | fold set before |
|---|---|
| `torch_bridge._fold_weight_chains` (the truth) | 160-op literal `elif` + `matmul` |
| `cost._folds_to_param` | `_FOLDABLE_ELEMWISE` (13) + `_FOLDABLE_VIEWS` (14) + `matmul`/`concat` = 29 |
| `egraph.extract._FOLDABLE_OPS` | `_FOLDABLE_ELEMWISE` + `matmul`/`concat` = **15 — no views** |

The elif list was widened to ~160 ops in the e6a618e semantic-coverage
pass and the cost side was never re-synced — `rsqrt` (the conv+BN
scale chain's root op), reductions, creators, the activation family
all missing.  Extraction's `_FOLDABLE_OPS` was worse: it predated
`_FOLDABLE_VIEWS` entirely, so a `reshape`/`transpose` node inside a
foldable weight chain billed the whole subtree at runtime — a
second, independent drift (`test_transpose_spelled_chain_does_not_pay`
pinned it as intended behavior; it now asserts the honest outcome).

## The fix — one table, three readers

`opmeta` gains the `fold-at-lowering` tag and the `_FOLD_AT_LOWERING`
membership (161 names = the lowerer's elif ∪ `matmul`), merged into
the registry at build so `ops(FOLD) → FOLD_OPS` is the projection.
All three consumers read it:

- `torch_bridge._fold_weight_chains`'s `elif` dispatches on
  `FOLD_OPS` — the literal list is gone, the *declined* family
  (slice/extraction views) is documented at the data site;
- `cost._folds_to_param` checks `FOLD_OPS` with the `matmul`/`concat`
  real-operand rule layered on (`_FOLD_EVAL_OPS` = the shallow-eval
  residue);
- `extract._FOLDABLE_OPS = FOLD_OPS` — the e-class discount now
  covers views and the whole widened set.

60 binding-only spellings (`sin`/`zeros`/`ones_like`/…) register into
the vocabulary through the table — `validate_aten_map` stays clean
(additive registration, no shadowing).  Semantics preserved:
`matmul` reaching the eval arm degrades identically (eval declines
Const/unresolvable args); every newly-foldable op was already folded
by the lowerer — only the *price* changed.

## Measured — the premise-pay leg, before/after

`lab/probes/conv_fold_premise_pay_probe.py` on the real eval-mode
`ConvBN` seed (same run, both legs):

| leg | before (`de72c64`) | after |
|---|---|---|
| transport | hooks rebuilt transitively | same |
| truth (real site) | 1eq/0ne accepted | 1eq/0ne — unchanged |
| truth (premise-anchored) | derivable, replayable, 2-step cert | same |
| **pay** | `fires=1` `changed=False` `verified=same` | `fires=1` **`changed=True` `verified=pass`** |
| `dag_cost` | spelled=34 874.3 vs fused=34 874.34 | spelled=34 874.3 vs **fused=17 438.7** |

The fused member now wins extraction by ~2× (the `batch_norm` eval
pass + every spelled weight op priced out; only `conv2d` remains).
The premise-pay floor evidence the retro named now exists: a
dependent mint that fires, truth-verifies, AND pays.

## The guarded sweep — verdict: still refuse, DEFAULT unchanged

`run_gauntlet` over `default_gauntlet_corpus()` (364 terms / 245
probes / 83 base rules) on the five floor candidates:

| rule | verdict |
|---|---|
| `conv_channel_{scale,scale_l,shift,shift_sub}_sp` (4 anchors) | REFUSED @ **truth**: `0 accepted / 2360 envs`, `derivable=False` — the enumeration bank cannot spell the conv channel-view corner, the real corpus has no spelled sites.  Same refusal as the earlier sweep — a domain-starvation gap the cost model cannot repair. |
| `batch_norm_expand` | REFUSED @ **typed-pay**: `fires=8 typed=8 paid=0` — spelling BN *out* adds ops on the corpus sites it fires on; the paying direction is the *composite* (BN onto conv), not the expansion. |

So the anchors remain **enabled-but-opt-in**: their dependent mint now
demonstrably pays, but their own admission still fails the truth leg
(guarded-region starvation — the named corner-reaching route is still
requirement #1).  `batch_norm_expand`'s enablement account is real
(its premise seat in the paying composite) but it pays nothing
itself.  The minted composite is `syn_*`-provenance — admission to
`DEFAULT` runs through the machine-admitted law machinery, not this
sweep.

## What stays open

- The conv corner-reaching truth route (enumeration vocabulary for
  `conv2d`/channel-views or witness-driven sites) — unchanged.
- `run_gauntlet`/`pl.measure` on guarded minted composites still does
  not terminate in the box (the 83-rule `_reach`/derivability
  saturation) — the composite was admitted via the bounded probe
  stages, not the full gauntlet.
- `FOLD_OPS` is the *torch lowering's* contract; another backend
  (ONNX sink, CUDA runner) folding differently would need its own
  alphabet — the tag's docstring scopes it accordingly.
- Pre-existing suite finding (not this change's):
  `test_every_conditional_rule_is_swept` counts 125 conditional rules
  vs the pinned 124 — a stale pin at HEAD, verified failing on clean
  `main` before this commit.

## Gates

ruff check/format clean on the shipped source; `ty` clean; vulture /
lint-imports / bandit / semgrep clean; radon ratchet's only deltas
pre-existing baseline drift (verified identical on stashed main).
Tests: `test_opmeta` projections + single-source pin,
`_folds_to_param` on the `rsqrt` weight spelling, `dag_cost` at
zero, `test_construction_gaps` retargeted; dev lane (`-m "not slow"`)
run logged in GOALS.
