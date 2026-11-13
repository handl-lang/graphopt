# cumsum carrier — the emitted-scan tier (`kind="seqs"`)

Date: 2026-11-14.  Scope: `mlopt_discovery.play` (`_cumsum_sites`,
`_all_scan_sites`), `mlopt_discovery.genkernel` (`scan_spec` /
`_scan_kind` / `_scan_info` / `_seqs_mode`, `seqs_denotation`,
`handler_denotation`, `_seqs_*` Triton emitters + bindings,
`scan_triton_bindings` module branch, `_seq_scan_spec`,
`__all__`), `tests/test_discovery_play.py` (`TestCumsumSites` +6),
`tests/test_discovery_genkernel.py` (`TestSeqsSpec` /
`TestSeqsDenotation` / `TestSeqsBindings` +16).

Question: `docs/reports/zoo_corpus_expansion.md` minted **zero
sites of every tier** on `_CumsumLinearAttention` — its scan is
spelled `cumsum(k ⊗ v)` over the time axis, a single op with no
`add(mul(·,·),·)` select-chain for the seq/seqm collectors to
read.  p37 (`p37_seqm_matrix_scan.md`) recorded "no honest fold
denotation exists" for cumsum carriers.  This report checks
whether that claim is still true and what the honest path is.

## Was p37's impossibility claim right?

**Half right — and the wrong half is now fixed.**  Read precisely,
p37 said two different things:

1. *Spelling boundary (true then, still true as a collector
   fact):* the raw-IR collectors only see unrolled
   `add(mul(·,·),·)` chains; a `cumsum` op is one node — no spine,
   no site.  That was and remains a collector limitation, not a
   denotation one.
2. *Denotation gap (real, now closed):* the existing `seq`/`seqm`
   minted ops denote the fold's **final state**
   (`seq_scan_denotation` returns `acc` after the last step; the
   seqm kernel stores only the last tile).  `cumsum` returns
   **every prefix state** — `out_t = acc_t` for all `t`.  No
   existing kind denoted that contract, so even a collector that
   saw the op could not have minted an honest site.  That WAS a
   genuine denotation gap.

The math itself was never the problem: a prefix sum IS a linear
recurrence — `S_t = S_{t-1} + x_t`, i.e. `a ≡ 1`, `b_t = x_t`,
zero carry.  p37's "different algebra" phrasing was about the
*chunked* forms (ChunkedSSD's `cumsum` + masked-attention chunk
algebra, ChunkedRetention's decay-masked quadratic) — those stay
out, unchanged.

## What was built — the `kind="seqs"` emitted scan

The smallest honest extension of the existing tier:

- **`play._cumsum_sites`** walks the term for
  `Op("cumsum"|"cumprod", child, dim=int)` nodes and mints
  `("applyd", ("aff_diag", A, B), "H", {"kind": "seqs",
  "axis": eff})` where `cumsum → (1, "X1")` with `H = Const(0.0)`
  and `cumprod → ("X1", 0)` with `H = Const(1.0)` — the scan's
  neutral element as a `Const` leaf, not a guessed parameter.
  `shape_of` normalises the step dim; `"chain"` is the cumsum node
  itself so `_route_scan_chains` rewrites it to the kernel call
  exactly as the unrolled chains are routed.  Registered in
  `_all_scan_sites`.
- **`genkernel.scan_spec`** accepts `kind="seqs"` (one-leaf
  `aff_diag` only, same rule as `seq`) and `_scan_info` records
  `info["scan_mode"]` from the RAW leaf specs: `"cumsum"` /
  `"cumprod"` / `"fold"`.
- **`seqs_denotation`** is the declared semantics — every prefix
  state.  For the closed-form modes it calls
  `torch.cumsum`/`torch.cumprod` directly: the referee's spelled
  arm MUST be the incumbent the delivery replaces (timing the
  kernel against a python-loop fold would grade it against a body
  nobody ships — the same trap `seq_scan_denotation`'s docstring
  warns about).  `"fold"` mode spells the generic emitted fold —
  `outs.append(acc)` per step, `torch.stack` moved back to the
  step axis.
- **`handler_denotation`** dispatches `seqs`; `gen_bindings`
  binds the denotation as the uncompiled fallback (delivery is
  never the ambient `applyd` map application).
- **Triton** (`scan_triton_bindings` now loads the generated
  *module* for seqs): closed-form modes emit TWO kernels —
  `_gs`, a **strided** `(P, T, K)` launch (rank-≤1 prefix, `K` the
  contiguous trailing block, real `tl.cumsum`/`tl.cumprod` along
  axis 0 — no transpose copy), and `_gsc`, the dense `(T, N)`
  fallback for shapes the strided view cannot express; `"fold"`
  mode emits the serial emitted fold (store `acc` inside the step
  loop — inclusive scan, any `T`).  Honest bounds: `T ≤ 256`
  (the `BT = next_pow2(T)` tile cap) and `dtype` must match —
  violations raise, the referee drops the claim.

## Verified

- **Denotation == incumbent, fp64 CPU:** `seqs_denotation` vs
  `torch.cumsum`/`torch.cumprod` — max abs diff **0.0**;
  fold-mode vs a hand-spelled emitted fold — **0.0**.
- **Kernels vs scan, fp64 CUDA:** strided cumsum **0.0**,
  cumprod **0.0**, axis-0 **0.0**, dense fallback on a
  non-contiguous operand **0.0**, fold mode ≤ ~1e-11 (accumulation
  order inside tolerance).
- **Zoo mint:** `CumsumLinearAttention`, `TaylorLinearAttention`
  and `ChunkedSSD` each mint exactly one `seqs` site
  (`aff_diag(1, X1)`, axis 1).  `ChunkedRetention` mints none —
  its cumsum bookkeeping is not a standalone scan op in the
  exported IR.
- **22 new tests** (`TestCumsumSites` 6, `TestSeqsSpec` 4,
  `TestSeqsDenotation` 5, `TestSeqsBindings` 7) — all pass under
  `scripts/pytest-capped.sh`; `test_discovery_play.py` 102 passed,
  `test_discovery_genkernel.py` 174 passed + 1 skipped
  (`-m "not slow"`).  `ruff check`/`ruff format --check` clean on
  both touched package files.

## Measured — does it pay?

The honest headline: **the carrier is now visible, denotable and
deliverable — and the measured-cost referee still declines it.**

Kernel vs incumbent on the zoo operand `(8, 32, 64, 64)` fp64
CUDA (the `(P,T,K)` strided launch, `BN=256` swept 64–1024):

| arm | time |
|---|---|
| generated `tl.cumsum` kernel | ~187 µs |
| `torch.cumsum` (the denotation/incumbent) | ~192 µs |

Parity-to-slight-edge on the raw kernel — but the referee times
 the *callable*, host wrapper included, interleaved against the
spelled arm: `PROFITABLE_LOG` recorded
`('scan_0', spelled≈211/200 µs, kernel≈230/205 µs)` — two
borderline windows, **both keep-votes False**.  `gen_probe` on
`_zoo_case(CumsumLinearAttention)` therefore offers **no claims**
(`gen_handlers == {}`), exactly the honest-decline contract: the
site mints, the spec is recognised, the Triton module binds and
verifies — and the claim is dropped because `torch.cumsum` is
already a tuned parallel scan (cub-class) and our tile kernel is
at parity, not ahead.

`TaylorLinearAttention`/`ChunkedSSD` mint the same site shape on
smaller tensors and are likewise dropped by `profitable`.

## The Blelloch angle — scoped honestly

`tl.cumsum`/`tl.cumprod` ARE the parallel-scan primitive (the
Blelloch/Hillis-Steele family inside the tile) — the denotation
exists (prefix sum is mathematically trivial) and the carrier is
implemented and verified.  Beating `torch.cumsum`'s own scan needs
either larger `T·N` (more columns per program amortises the
tile setup), a decoupled-lookback multi-kernel scan for `T > 256`,
or int32-offset fast paths — all real engineering, none claimed
here.  What IS claimed: the spelling boundary is gone, the
fold-vs-emitted contract is now explicit in the tier grammar, and
the pay/no-pay answer is *measured* per site rather than assumed.

## Honest boundaries

- **`T ≤ 256`** for the closed-form kernels (`BT` tile cap); the
  serial `"fold"` mode has no cap but will not beat `cumsum`.
- **Strided view needs prefix rank ≤ 1** and C-contiguous
  trailing dims; anything else takes the dense `(T,N)` path —
  correct, but it pays a ~190 µs fp64 transpose copy that loses
  to the incumbent.  Multi-dim prefixes (e.g. `(B,H,T,d)` axis 2)
  always take the dense path.
- **`dtype` must match** the promoted scan dtype — a mismatch
  raises (decline), never silently casts.
- **Same-spec dedup:** two `cumsum` sites with identical spec
  canon collapse to one handler (the `_canon_taken` rule the seq
  sites already live under) — the first occurrence routes.
- **`ChunkedSSD`'s chunk algebra stays out**: only the raw
  `cumsum` op inside it is covered; the masked-attention chunk
  recurrence is still not a carrier fold.
- **`ChunkedRetention`** shows a pre-existing `gen_probe`
  `IndexError` at `d=64 t=32` unrelated to this change (it mints
  no seqs sites — verified by census).

## Files

- `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `_cumsum_sites`, `_all_scan_sites` (+1 collector).
- `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py` —
  `__all__`, `scan_spec` docstring + `leaves` plumbing,
  `_scan_kind`/`_scan_info`/`_seqs_mode`, `seqs_denotation`,
  `handler_denotation`, `_seq_scan_spec`, the `seqs` codegen
  section (`_SEQS_BN`/`_SEQS_MAX_T`, `_seqs_triton_source`,
  `_seqs_scan_source`, `_seqs_fold_source`, `_seqs_strided_view`,
  `_seqs_dense_args`, `_seqs_binding`, `_seqs_scan_binding`,
  `_seqs_fold_binding`), `_scan_triton_source`/`scan_triton_bindings`
  seqs branches.
- `tests/test_discovery_play.py` — `TestCumsumSites`.
- `tests/test_discovery_genkernel.py` — `TestSeqsSpec`,
  `TestSeqsDenotation`, `TestSeqsBindings`.
