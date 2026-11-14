# Low-memory verify — `lower(verify=True)` on 4 GB

Follow-up to `whisper_experiment_r2.md` wall "verify-time weight
duplication (turbo)".  whisper-turbo bf16 searches fine (163–213 s)
but `lower(verify=True)` OOM'd on the RTX 2050 4 GB: the verify gate
materialised a fresh reference lowering next to the delivered module
— two full weight sets (~2 × 1.6 GB bf16) plus activations > ~3.7
GiB usable, so the largest checkpoint could only deliver *unverified*.

## What actually duplicated

`orchestrator.lower()` (eqsat) builds the delivered module, then for
verify calls `sink.lower(result.ir, params)` — a second
`IRModule` over the *same* parameter values.  Inside
`IRModule._build_params` every registered param is `.clone()`d, and
`_align_to` lazily lifts the live tables (`_param_map` /
`_param_values`) to the inputs' device at first forward.  So at
verify time the device held:

- delivered module's live weights (~1.6 GB, GPU after its forward),
- reference module's live weights (~1.6 GB, GPU during its forward),

while `params` (`SearchResult.param_values`) and each module's
registered `_parameters` clones stay host-resident.  Nothing was
*wrong* — the check just needed both weight sets simultaneously.

## Fix: park, don't weaken

The equivalence check is unchanged — same two forwards, same clones,
same `verify_equiv` metric.  Only device residency between the
forwards changed:

- `IRModule._park_params()` (torch_bridge.py) moves the module's
  *live* weight tables to CPU and resets `_dev_key`, so the next
  forward re-aligns exactly as a fresh module would.  The CPU
  replacement prefers `self._parameters[k]` — the registered
  attribute `_build_params` installed before any `_align_to` lift,
  which `_align_to` never rewrites — so the park **rebinds to an
  existing host clone instead of allocating a second one** (the park
  trades VRAM for RAM; a fresh `.to("cpu")` per tensor would briefly
  hold both).  Live tensors with no CPU registered counterpart fall
  back to a real `.to("cpu")`.
- `report.verify_module()` calls `_park_params` on whichever side
  offers it (duck-typed — only `IRModule` implements it): both sides
  park before the first forward (the delivered module may already be
  device-resident, e.g. after a compile runner), the reference
  re-parks before the optimized forward, and each parked module
  re-aligns on its next forward.  Modules without the hook keep the
  historical both-resident behaviour — arbitrary user modules passed
  as `ref` (autotune's `sink.verify(model, …)`) are never moved.

Verification strength is identical; what changed is the memory
profile: the pair never holds two full weight sets on the compute
device at once.

## Measured (RTX 2050 4 GB, synthetic `matmul` pair, ~1.1 GiB weights each)

| arm | peak `max_memory_allocated` |
|---|---|
| old both-resident forwards | **2.19 GiB** |
| `verify_module` (parked) | **1.10 GiB** |
| GPU-resident entry (delivered pre-forwarded), parked verify | 1.10 GiB |

End-to-end (`bench/drivers/probes/whisper_probe.py opt encoder
<model> <dtype> 100000 cpuparams`, RTX 2050 4 GB):

| arm | result |
|---|---|
| small fp32 | search 27.3 s; `lower+verify` 0.8 s **verified=True**; delivered 223.4 ms (matches r2's 225–228 ms — no regression) |
| small bf16 | search 30.3 s; `lower+verify` 0.4 s ran to completion — **verified=False, measured `max_rel=0.031`**: the pointwise gate rejects bf16 reassociation at `rtol=1e-4` (bf16 carries ~8 mantissa bits; fp32 passes the same gate). The verify *ran* — the memory wall is what fell. bf16 deliveries need a dtype-appropriate tolerance, a separate question from memory. |
| turbo bf16 | search 174.3 s; **`lower+verify` 1.9 s — verify runs to completion on 4 GB, no OOM** (was: hard OOM in r2). `verified=False` — same bf16-vs-`rtol=1e-4` tolerance story as small bf16, not a memory or correctness failure. Delivered 312.8 ms. `CudaGraphRunner` still `cuda_graph=False, fail=oom` (capture needs headroom the resident weights don't leave — unchanged, expected). |

## The remaining wall: host RAM, not VRAM

Verify no longer duplicates weights *on the device*, but the pair
still materialises ~3 host-resident copies of the weight set:

- `SearchResult.param_values` / the `params` dict lower() passes
  (~1.6 GB for turbo bf16),
- the delivered `IRModule`'s parameter clones (~1.6 GB),
- the reference `IRModule`'s parameter clones (~1.6 GB),

plus interpreter + CUDA host overhead — ~6.5–7+ GB against ~4–5 GB
actually available on this 11 GB box (earlyoom was killing unrelated
desktop processes in the same window).  The GPU side is fixed; the
next bounded step is param-storage sharing across the verify pair —
both lowers receive the *same* `params` dict object, so
`IRModule._build_params`' `.clone()` is the remaining duplication
point.  Skipping the clone (or a per-sink clone cache keyed on the
source tensor's identity) would collapse the pair to ~1× weights, but
it changes the "delivered module owns its weights" aliasing contract
and is deliberately left as follow-up work.

## Tests

- `tests/test_reports.py::test_verify_module_parks_parkable_modules_between_forwards`
  — ordering spy: park both → ref fwd → park ref → opt fwd; non-parkable
  pairs keep the old order.
- `tests/test_torch_bridge_edges.py::test_park_params_cpu_residency_and_realign`
  — CPU no-op park keeps the `_param_values`→`_param_map` aliasing
  invariant and re-evaluates identically.
- `tests/test_torch_bridge_edges.py::test_park_params_moves_gpu_tables_to_cpu`
  (CUDA) — the park moves the *live* tables (not the stale registered
  clones) and the next forward re-aligns to the input device.
- `tests/test_torch_bridge_edges.py::test_verify_module_lowmem_never_holds_both_weight_sets`
  (`requires_cuda`, skips >8 GiB GPUs) — each side's weights ≈28 % of
  free VRAM; asserts verify passes and peak < 0.48×free (two resident
  sets would peak ≈0.56×free).

## Honest limits

- **bf16 verify verdicts.**  The memory wall fell, but the pointwise
  gate (`rtol=1e-4`) rejects bf16 deliveries on *tolerance*, not
  memory: whisper-small and whisper-turbo bf16 both verify-run and
  report `verified=False` (bf16 reassociation across the rewritten
  graph exceeds 1e-4 rel; fp32 passes the same gate).  bf16 callers
  need a dtype-appropriate `rtol`/`atol` or the task-metric gate —
  orthogonal to this fix, flagged for follow-up.
- An earlier turbo attempt was killed by the host's `earlyoom`
  daemon during module materialisation — host RAM (~3 weight copies,
  see above) is a real secondary constraint on this 11 GB box; the
  successful rerun had ~7 GB available.  On a roomier host the
  triple-residency is merely wasteful, not fatal.

- The park trades VRAM for *transient* host RAM only when live weights
  are device-resident and no CPU registered counterpart exists —
  for `IRModule` the registered clones already cover this, so parking
  allocates nothing.
- Carrier executors (`BatchedExecutorBase` families) and
  runner-wrapped modules do not expose `_park_params`; a verify pair
  involving them keeps the old residency.  The generic `IRModule`
  path — the whisper case — is covered.
- `_task_verify` (orchestrator, task-contract verifies) runs its own
  forwards outside `verify_module`; a task-gated verify of a
  device-resident pair can still co-reside.  Not hit by the whisper
  probe (no task contract).
- `IRModule`'s registered `_parameters` entries remain the pre-align
  CPU clones for the module's lifetime (~1× weights of host RAM per
  module) — pre-existing behaviour, unchanged by this fix.
