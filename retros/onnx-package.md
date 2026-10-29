---
title: 'Retro — catopt-onnx: 75% → 100% coverage in one climb'
type: retro
date: 2026-10-14
---

# Retro — catopt-onnx: 75% → 100% coverage in one climb

## The arc

`catopt-onnx` landed as a deliberately-unfloored package (the sink
in `f3b43f1`, the source in `32fed70`) — measured honest gap at
climb start: **75%** statement+branch coverage, with `liftmap.py`
(the ONNX→IR op table) at 40% and `lift.py` at 79%.  The climb ran
in two files mirroring the modules they cover:

| file | scope | result |
|---|---|---|
| `tests/test_onnx_liftmap.py` (~110 tests) | `liftmap` per-op minted spellings, attr folds, composed lifts and every declared refusal; `lift.py` engine edges | liftmap 40→100, lift 79→100 |
| `tests/test_onnx_opmap.py` (~40 tests) | `opmap` alternative attr spellings and refusals; `lower`/`arrays`/`report` edges | all four → 100 |

Helpers reuse `tests.test_onnx_source`'s model builders; the new
seam is `_lifter(...)` driving `GraphLifter` directly with a partial
`value_info` table — the same object `_collect_value_info` produces
when ONNX's own inference leaves an intermediate unshaped (an
honest input of the class, not a fake).  Final measured number:
**100% — 1673 statements, 514 branches, zero misses, zero
suppressions.**  The package joins the `fail_under=100` wall (not a
separate ratchet — the measured number IS the wall).

## What the climb proved (beyond lines)

Three real defects surfaced — two fixed in the same change, one
documented as an upstream deviation:

1. **Non-default-domain nodes were not refused.**  Both docstrings
   (`liftmap.ONNX_LIFTS`, `lift_model`) claimed a node with a
   non-empty `domain` "refuses through the missing-entry path" — but
   `_lift_node` never checked `node.domain`, so an
   `ai.onnx.ml`-domain `Relu` silently lifted as default-domain
   `relu` (a vendor `Add` could be *anything*).  Fixed with a
   domain guard in `_lift_node`, mirrored in `_scan_constants` so a
   foreign `Constant` can't slip a leaf into the environment.
   Pinned by `test_a_non_default_domain_node_is_refused`.
2. **`_emit_args` typed scalar constants by the *first* sibling.**
   `where(gt(x,a), x, 0.0)` — exactly what the `ThresholdedRelu`
   lifter mints — picked the bool predicate's dtype for the `0.0`
   constant and emitted an invalid `Where(bool→float mix)` graph.
   Now a scalar adopts the first *non-boolean* sibling dtype (a
   predicate is never the constant's own type), falling back to the
   first tracked dtype for all-bool terms.  Pinned by the
   thresholded-relu round trip.
3. **`remainder` lowered to an unrunnable graph for int dtypes.**
   `Mod(fmod=0)` on `int64` inputs mints `remainder`, which the sink
   composed as `x - floor(x/y)*y` — but ORT has no integer `Floor`
   kernel either.  `Mod(fmod=0)` IS the integer path the runtime
   requires, so `_remainder` now emits it for `i`/`u`-kind dtypes
   and keeps the composition for floats.  Pinned by the
   `mod-int` round trip.

**Upstream finding (not a bug, a deviation):** ONNX's spec and its
reference evaluator say `Gather` with a scalar index drops the axis
(`(4,)`); onnxruntime's CPU EP keeps a `(1, 4)`.  The lifter is
spec-faithful — `test_gather_scalar_index_paths` verifies the
lifted program against NumPy `w[2]` rather than ORT's deviation.

## Reachable-by-construction arcs the climb exposed

Two branches looked dead and weren't — the honest reads:

- `lift.py` `_const_leaf`'s param cache (`params.get` hit): the
  `_env` memo means `term()` calls it once per name, but the
  graph-*output* path resolves leaves without the memo — a model
  listing the same constant name as two outputs re-enters the
  materialiser.  `test_a_twice_named_constant_output_materialises_once`
  covers it.
- `lift.py` `_lift_node`'s `if name:` filter on bound dicts: no
  shipped lifter produces an empty-name key, but the guard is the
  table's contract for *registered* lifters — exercised by
  registering a lifter that binds `""` (`monkeypatch.setitem` on
  `ONNX_LIFTS`, the documented extension seam) and asserting the
  env stays clean.

Nothing needed a `pragma: no cover` — the wall is honest.

## Honest seams (documented, not fakes)

- `GraphLifter(graph, initializers=..., value_info={})` — drives
  the "not statically known" refusals (squeeze/unsqueeze/split/
  gather/conv/layer-norm/global-pool/CastLike) through the same
  partial table `_collect_value_info` produces when inference can't
  type an intermediate.
- `_raw_model(...)` — builds the same `ModelProto` minus
  `onnx.checker`, for deliberately ill-formed-but-constructible
  graphs (extra outputs, empty output names, duplicate output
  names) that an exporter could still emit and the lifter must
  still answer.

## Floor statement

`catopt_onnx` reports in the main `coverage report` wall
(`fail_under=100`, `--omit="*/catopt_discovery/*"`); discovery keeps
its own `--fail-under=99` ratchet.  The ONNX tests importorskip
`onnx`/`onnxruntime` — the package declares both as dependencies,
so the floor assumes the declared environment (same convention as
the CUDA-marked tests).
