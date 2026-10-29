# P23 — M1 closed: eight variants generated from terms, all gated, all matching

Probe: `bench/drivers/probes/awq/m1_verdict.py`; JSON `p23_m1_verdict.json`.
Engine: `compile-triton` (Rust, `Term` → Triton). IR: `lib-kernel-ir`.

## The definition of done, item by item

The build list asked for four `split_k` variants generated **from IR
terms**, agreeing with `awq_gemm` at cosine ≥ 0.9999, at least one matching
the hand-written kernel, on two opposite geometries.

| shape | split | generated | hand | gen/hand | cosine vs `awq_gemm` | % roof |
|---|---|---|---|---|---|---|
| `down_proj` 896×4864 | 1 | 149.9 | 149.9 | 1.00× | 1.000000 | 16.4 |
| | 2 | 79.1 | 79.2 | 1.00× | 1.000000 | 31.1 |
| | 4 | 54.4 | 54.3 | 1.00× | 1.000000 | 45.2 |
| | 8 | **52.3** | 52.6 | 0.99× | 1.000000 | **47.0** |
| `gate_proj` 4864×896 | 1 | 66.1 | 66.3 | 1.00× | 1.000000 | 37.2 |
| | 2 | 54.5 | 54.2 | 1.00× | 1.000000 | 45.1 |
| | 4 | **52.4** | 52.0 | 1.01× | 1.000000 | **47.0** |
| | 8 | 56.0 | 56.1 | 1.00× | 1.000000 | 43.9 |

- 8/8 variants at **cosine 1.000000** against the incumbent.
- **Every** variant within 1 % of the hand kernel (tolerance was 5 %).
- No generated source contains an atomic (P13's retraction is structural).
- `split_k` matters and the search can see it: 149.9 → 52.3 µs on
  `down_proj` (2.9×) purely by choosing 8 partitions instead of 1.

## What this does *not* prove

The lowering emits the same two-stage structure the hand kernel uses, so
"generated matches hand" is the expected outcome — the content is that the
**IR → source path is faithful**, not that codegen found something new.
A fixed lowering that only reproduces one kernel will not beat it either:
M1's withheld question was whether terms are good enough to *search*, and
the honest answer so far is "they are good enough to reproduce".

Two concrete limits, stated so they are not discovered later:

1. **47 % of roofline is still the number** (P22: hand kernel 58 % at the
   `bn=32, split=4` config; this probe pinned `bn=64` across splits for
   comparability). Marlin is at 86–90 %. The gap is the unpack arithmetic,
   not the IR.
2. **The IR cannot express scheduling.** No pipelining, no shared-memory
   residency, no accumulator reuse across k-tiles — the review's objection
   to the nine-op vocabulary stands, and it is exactly the missing 40 %.

## Artifacts

| item | where | state |
|---|---|---|
| ADR | `host/project/adrs/0165-kernel-ir-is-term-plus-ops.md` | written (number 0165; 0160 was taken) |
| `lib-kernel-ir` | new repo, 21 tests | ops + denotations + `Term` + JSON boundary |
| `compile-triton` | new repo, 6 cargo tests | reads the term, refuses non-`qgemv` and non-IR ops, emits |
| equality gate | `bench/benchkit/equality.py`, 6 tests | external incumbent, refuses agreement-on-a-mistake |
| search driver | `kernelopt/src/mlopt_kernel/search.py`, 12 tests | gate → bound → measure, cost never ranks |
| M1 verdict | this report | **PASSED** |

**Kill criterion, not triggered.** The pre-registered condition was that if
a fixed lowering could not match hand-written Triton, generalising would be
wasted work. It matches — so M2 (a second operator, a schedule vocabulary,
search at scale) is defensible. It is also still unproven: nothing here
beats the hand kernel, and the 43-point gap to Marlin lives in the schedule
the IR cannot yet state.
