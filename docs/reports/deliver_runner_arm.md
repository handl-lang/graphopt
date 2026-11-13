# Deliver-path runner arm — `CudaGraphRunner` inside `_torch_deliver`

**Date:** 2026-11-15 · **Scope:** `mlopt_discovery.play._torch_deliver`
(the `torch` and `gen` domain deliver hooks, `--deliver` CLI arm).

## What landed

`_torch_deliver` used to stop at lower + verify.  It now runs a
runner arm after a **passed** verify
(`packages/mlopt-discovery/src/mlopt_discovery/play.py`):

- `_is_plan_tape` (`:390`) — reports whether the lowered module is
  the dispatch-bound shape: an `IRModule` with a built `_EvalPlan`
  (`_root_plan is not None`), or a batched executor exposing
  `capture_cuda_graph`.
- `_max_abs_delta` (`:410`) — replay-vs-eager output delta over
  tensors or (nested) tuples of them; NaN for non-tensor outputs.
- `_deliver_runner_arm` (`:436`) — the arm itself; `_torch_deliver`
  merges its keys only when `rep.passed` (`:561-562`).

## Behaviour contract

- **Device is the case's contract.**  Capture is attempted only when
  *every* example input is already a CUDA tensor.  A CPU deliverable
  is never silently promoted to CUDA — the arm declines with
  `runner_note: "inputs not CUDA tensors"`.  (Promoting would
  change the artifact's device relative to what `verify` checked;
  that is a product decision for the caller, not the hook.)
- **Opt-in-by-availability, auto-with-fallback.**  `mlopt-cuda` is an
  optional dep (absent → `"mlopt-cuda not installed"`); a declined or
  raising capture → `cuda_graph: False` + `runner_note` (the runner's
  own `stats["cuda_graph_failure"]` classification is surfaced when
  present).  Capture failure can never fail a deliver — every exit
  path returns the eager `module`.
- **Honest timing.**  On capture, both arms are timed with the same
  `_time_module` median (`reps=50`, device-synchronised on CUDA):
  `deliver_us` is the eager plan-tape module, `graph_us` the replay
  wrapper.  The static-input copies are *inside* the timed call —
  `_GraphReplayModule.forward` does `buf.copy_(t, non_blocking=True)`
  then `graph.replay()` per call, and calls whose arity/shape/dtype/
  device differ from the capture signature fall back to the wrapped
  eager forward internally (`mlopt_cuda/runners.py:117-134`).  The
  copy cost and the shape-baked contract are measured, not hidden.
- **Numeric check before delivery.**  `graph_max_abs` compares
  `module(*xs)` vs `wrapped(*xs)` under `no_grad` — the same
  verification approach as `gen_probe._probe_graph_arm`
  (`play.py:~2010`).  `module` in the result is overridden with the
  `_GraphReplayModule` only when capture *and* the post-capture
  timing/compare block both succeed.

Result keys (additive; `module` may be the replay wrapper):

```python
{"runner": "cuda_graph", "plan_tape": True, "cuda_graph": True,
 "deliver_us": 428.8, "graph_us": 96.6, "graph_speedup": 4.437,
 "graph_max_abs": 0.0, "module": <_GraphReplayModule>}
```

On decline: `cuda_graph: False`, `runner_note: <reason>`, no timing
keys, `module` stays the eager `IRModule`.

## Measured — the known case, end to end

`zoo:GatedDeltaRule`, `zoo.zoo(64, 32, 8)`, `_zoo_case(dtype="fp32")`,
model + feed moved to CUDA, `play.deliver("gen", case,
ScriptedPlayer(), budget=12)` on the RTX 2050 (this session,
serial):

| arm | µs/call |
|---|---|
| delivered `IRModule` (eager plan tape) | 428.8 |
| `CudaGraphRunner` replay | 96.6 |

`graph_speedup` 4.44×, `graph_max_abs` 0.0, `verified`/`cert` true.
The delivered `module` is a `_GraphReplayModule` wrapping the
verified `IRModule` and answers `m(feed)` on CUDA.

This is consistent with `t034_module_costing.md` (graph arm 114.8 µs
= 1.53× compile fp32 in the `gen_probe` context) — the absolute
numbers differ because the deliver path mints/plays a different
extraction under a different budget; the qualitative result is the
same: the runner removes the ~350 µs host-dispatch share the tape
interpreter pays per call.

CPU path, same case un-moved: `cuda_graph: False`,
`runner_note: "inputs not CUDA tensors"`, `plan_tape: True`, eager
module delivered — clean decline, deliver still verified.

## Tests

`tests/test_discovery_play.py`:

- `test_deliver_runner_arm_cpu_declines` — CPU torch-domain deliver
  asserts the decline keys and that no timing keys appear.
- `test_deliver_runner_arm_cuda_captures` (`requires_cuda`) — the
  GatedDeltaRule fp32 CUDA zoo case above; asserts capture, zero
  output delta, and `graph_us < deliver_us`.

`pytest tests/test_discovery_play.py tests/test_domain_spec.py`:
163 passed (memory-capped, serial).  `ruff check` +
`ruff format --check` clean on `play.py`; `ty check` clean.
CUDA-only bodies carry `# pragma: no cover` per repo convention.

## Caveats / follow-ups

- **Shape baking.**  The replay wrapper serves exactly the
  capture-time signature; other shapes fall back to eager *inside*
  the wrapper — correct but the timing keys describe only the
  captured signature.
- **Static output.**  `wrapped(*xs)` returns the graph's static
  output buffer, overwritten by the next replay — clone to keep it
  (documented on `_GraphReplayModule`).
- **Multi-input feeds.**  `_torch_deliver` verifies with
  `(ex["input"],)`; for tuple feeds that nests the tuple as a single
  arg (pre-existing behaviour — such cases currently fail verify
  before the arm runs).  The arm itself normalises tuple feeds
  correctly for the runner; fixing the verify call is a separate
  change.
- **`--deliver` CLI** prints the dict minus `module`, so the new
  keys surface in the CLI output automatically; there is no CLI flag
  to force-disable the arm (declines are already honest and cheap).
- The working tree also carries unrelated uncommitted work
  (`_route_scan_chains` deliver-route in `meta_arena.py` /
  `genkernel.py` / `play.py` scan-site `"chain"` keys,
  `test_discovery_genkernel.py`) — not part of this change; left
  untouched, nothing committed.
