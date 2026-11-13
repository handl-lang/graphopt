# p37 — seqm: matrix-state scans with state feedback; the delta-rule family mints, verifies, and pays

Date: 2026-11-13.  Scope: `mlopt_discovery.play` (`_seq_step_cands`
add-form, `_seq_deindex` *state* argument, `_seqm_site` /
`_seqm_scan_sites`, `_all_scan_sites`), `mlopt_discovery.genkernel`
(`_seqm_spec` recogniser, `seqm_denotation`, `handler_denotation`
dispatch, the `_seqm_*` Triton emitter + lazy per-signature binding,
`scan_triton_bindings` branch, `_site_parts` axes plumbing),
`tests/test_discovery_play.py` (`TestSeqmSites` +6),
`tests/test_discovery_genkernel.py` (`TestSeqmSpec`/`TestSeqmDenotation`/
`TestSeqmBindings` +14).  Motivation: p36's recorded boundary —
"genuinely non-diagonal and chunked recurrences remain out of
scope".  The named targets were Mamba's real recurrence form and
the delta-rule family (`h_t = A_t h_{t-1} + B_t x_t` with matrix
state).

## Census — what non-diagonal recurrence shapes exist in the zoo

Method: per-workload intake IR (`TorchSource`, fp64), spine
candidates from `play._seq_spines` (len ≥ 2), old-grammar decline
reasons classified, plus op histograms for `select`/`matmul`/
`cumsum`/`einsum`.  Results (zoo 16×8×2 fp64):

| workload | recurrence shape | verdict |
|---|---|---|
| DeltaRuleBlock | `S += β·k⊗(v − kᵀS)`, state `(B,dk,dv)` — state as a DIRECT `add` operand, rank-1 write, `kᵀS` matvec read | **seqm site minted** (was: 12 spines, all `deindex-a` under old grammar) |
| GatedDeltaRule | `S ← αS + βk⊗(v − αkᵀS)` — same skeleton plus scalar gate `a_t` and in-residual gate `s_t` | **seqm site minted** (was: 12 spines, all `skeleton-mismatch`) |
| MambaSelectiveScan | diagonal decay + outer-product write | unchanged — `seq` site via p36 view-lift, still pays |
| GatedLinearAttention | `gate⊙S + k⊗v` | unchanged — `seq` site, still pays |
| RealGatedLRU | diagonal recurrence | unchanged — `seq` site, still pays |
| ChunkedSSD | chunked/parallel form: `cumsum` + masked attention + matmul | **out** — a parallel scan, not an unrolled affine fold; no honest fold denotation exists without a chunk algebra proof |
| ChunkedRetention | decay-masked quadratic attention | **out** — not a carrier fold at all |
| TaylorLinearAttention | `cumsum`-based (2 cumsum / 3 einsum nodes) | **out** — cumulative-sum carrier, different algebra |
| RWKVMixBlock | select/add chains present (8 spines) but no state-feedback step reads | **out** — `_seq_deindex` declines every candidate honestly |

Everything else in the zoo: no select-chain spines ≥ 2.

## The tier — `kind="seqm"`: `acc = a_t·acc + b_t(acc)`

The diagonal `seq` grammar is `add(mul(a_t, h), b_t)` with
state-free operands.  The delta-rule spelling breaks BOTH halves:
the state is a direct `add` operand (`S + write`, unit map) and
the write reads the accumulator back through ONE
`squeeze(matmul(K, S))` projection.  The extension is deliberately
the smallest honest slice:

- `play._seq_step_cands` gains `(None, c0, c1)` / `(None, c1, c0)`
  readings for `add` nodes — `None` marks the implicit unit map.
  Only `_seqm_site` accepts them; `_seq_site` declines `None`.
- `play._seq_deindex` gains a `state` argument: an occurrence of
  the step's own state factor inside the step expr abstracts to
  the `"H"` metavar instead of recursing into the previous chain
  link.  **Bug found by the new tests:** the check must be `==`,
  not `is` — `Op.make` hash-conses ops but `Var`/`Param` leaves are
  plain frozen dataclasses, so a sibling recurrence's equal-but-
  distinct `Var("H0", …)` interns the same `matmul` node while the
  spine's state factor stays the other object; `is` silently
  missed and the site declined (order-dependent — gated built
  first broke plain).  Structural equality is also the sound
  reading: an equal term denotes the accumulator's value.
- `play._seqm_site` mints `("applyd", ("aff_diag", a, b), "H",
  {"kind": "seqm", "axis": …})` only when: every step's `a`/`b`
  deindex to identical skeletons, `a` is accumulator-free, `b`
  CONTAINS `"H"` (state feedback is the point — a state-free `b`
  is the plain seq form's job), each operand's own select dim is
  in range with extent == step count, and select indices walk
  positionally.  There is no shared broadcast frame — matrix-state
  operands live in different spaces (`k ∈ R^dk`, `v ∈ R^dv`,
  `acc ∈ R^(dk,dv)`) — so the site records a PER-OPERAND `axes`
  map that flows through `_site_parts` into the handler entry.
- `genkernel._seqm_spec` recognises exactly one `b` skeleton:
  `mul(C, unsqueeze(sub(V, [mul(S,)] squeeze(matmul(K, H), dim)),
  dim))` — the rank-1-feedback write `C ⊗ (V − [S·]KᵀH)` with
  either `mul` order, `squeeze` dim ∈ {1, −2}, `unsqueeze` dim ∈
  {−2, 1}.  Everything else declines: `H` outside that position,
  `H` inside `a`, non-elementwise sub-specs (checked through the
  same `_triton_expr_h` renderer the kernel uses — spec and
  delivery cannot drift apart).
- `genkernel.seqm_denotation` is the delivery-independent spelled
  fold in plain torch: per operand it `movedim`s ITS OWN axis to
  the front, then walks `t` evaluating `a_t`/`b_t` through
  `_eval_spec` against `{metas: step slices, "H": acc}` —
  `matmul(K, acc)` included — exactly the chain's order.
  `handler_denotation` dispatches on `info["kind"] == "seqm"` and
  requires the handler's `axes` map (raises `ValueError` without
  it — no guessed shared axis).

## Codegen — a real tiled kernel, not a pretend-flat one

`scan_triton_bindings` routes `kind="seqm"` to `_seqm_binding`,
which resolves a per-call signature (`_seqm_signature`) from the
bound operand shapes and JITs a `_gsm` kernel per signature —
source hashed into the kernel cache (`_gsm_<name>_<digest>.py`,
inspectable on disk), the loaded `triton.jit` callable memoised in
`_SEQM_LOADED` (the first cut re-imported the module per call —
~3 ms of pure overhead, fixed before any claim priced).

The kernel: one program per `(batch, dv-block)` owns a
`(BLOCK_DK, BLOCK_DV)` tile of the `(dk, dv)` state; the `t` loop
is serial.  Each elementwise operand loads under its skeleton
role — `frame` operands normalise to the `(DK, DV)` tile
(fscalar/fcol/frow/full), `row` operands stay `(BV,)` vectors,
the matvec key `K` loads as a `(DK,)` vector — then
`proj = Σ_i K_i·acc_ij` (masked `tl.sum` matvec),
`r = V − [S·]proj`, `acc = a·acc + C·rᵀ`.  `dk ≤ 64`
(`_SEQM_MAX_DK`), `dv` tiles freely; shape ineligibility raises —
the same honest-decline contract the flat seq binding keeps, and
the referee drops the claim rather than shipping a wrong kernel.

## Soundness — before any claim priced

- Fold vs unrolled chain, `handler_denotation` vs `eval_term`,
  fp64: **max abs diff 0.0** on DeltaRuleBlock and GatedDeltaRule
  zoo intake, and on the synthetic delta/gated chains in
  `TestSeqmSites` (plain torch, CPU).
- Triton kernel vs fold denotation, fp64 CUDA: DeltaRuleBlock
  **5.0e-16**, GatedDeltaRule **2.2e-16**; synthetic spec
  **≤ 8.7e-19** across shapes.
- End-to-end `gen_probe` on the real zoo path: both workloads
  `delivered=True, verified=True` (DeltaRuleBlock max_abs 0.0,
  GatedDeltaRule 2.2e-16).

## Measured — does it pay?

Synthetic spec sweep (fp64 CUDA, kernel vs spelled torch fold,
steady-state after warmup):

| B | T | dk | dv | kernel | fold | ratio |
|---|---|---|---|---|---|---|
| 2 | 4 | 16 | 16 | 126 µs | 250 µs | **0.50×** |
| 2 | 16 | 16 | 16 | 133 µs | 949 µs | **0.14×** |
| 2 | 64 | 16 | 16 | 129 µs | 3599 µs | **0.036×** |
| 4 | 64 | 32 | 32 | 175 µs | 3626 µs | **0.048×** |
| 2 | 128 | 32 | 32 | 328 µs | 7204 µs | **0.046×** |
| 2 | 64 | 64 | 64 | 506 µs | 3592 µs | **0.14×** |

The kernel is ~flat in T (one launch, serial step loop, tiled
state); the spelled fold is a T-step python loop — the margin
grows with T.  First-call JIT compile is a one-time ~90–460 ms,
amortised across calls and cached on disk per signature.

Real zoo path (`_zoo_case` → `profitable`, fp64 CUDA 16×8×2,
reps=20):

- **DeltaRuleBlock `scan_4`**: spelled fold 333.6 µs vs Triton
  149.5 µs → **ratio ≈ 0.45×, KEPT**.
- **GatedDeltaRule `scan_8`**: spelled fold 346.7 µs vs Triton
  173.1 µs → **ratio ≈ 0.50×, KEPT**.

Both are the ONLY seqm sites zoo-wide — exactly the two
delta-rule workloads the census named.

## Honest boundary (what seqm is NOT)

- **One skeleton.**  `b` must be `C ⊗ (V − [S·]KᵀH)` — a rank-1
  write whose only state read is ONE `matmul(K, H)` projection.
  A general matrix `A_t·H` (true matrix-state `h' = A_t h + b`)
  has no elementwise `b`-skeleton and declines at `_seqm_spec`.
  Mamba's real `(B,d,n)` recurrence is covered only through its
  diagonal sub-chain (the p36 `seq` site), not as a matrix state.
- **`dk ≤ 64`** register-tile cap; `dv` unbounded (grid-tiled).
  State must be exactly `(B, dk, dv)`; operands `(T, B, *rest)`
  after per-operand `movedim`, rests rank ≤ 2.
- **Chunked/parallel scans stay out.**  ChunkedSSD's `cumsum` +
  masked-attention carrier and ChunkedRetention's decay-masked
  quadratic form are different algebra — treating them as unrolled
  folds would be unsound, and nothing here claims them.
- **cumsum carriers stay out** (TaylorLinearAttention): a
  cumulative sum is a scan with `a=1`, but its spelled form is a
  single `cumsum` op, not an unrolled select chain — no spine,
  no site, honest zero.
- The kernel pays launch + per-step operand materialisation;
  at very small T·dk·dv the spelled fold can still win — the
  `profitable` referee decides per site, as designed (and did:
  both zoo sites kept, nothing else offered).

## Files

- `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `_seq_step_cands` add-form, `_seq_deindex` `state` param
  (equality, not identity), `_seqm_site`, `_seqm_scan_sites`,
  `_all_scan_sites`.
- `packages/mlopt-discovery/src/mlopt_discovery/genkernel.py` —
  `_seqm_spec`/`_seqm_parse_b`/`_seqm_parse_p`/`_seqm_sub_ok`,
  `seqm_denotation`, `handler_denotation` seqm branch,
  `_seqm_signature`/`_seqm_load_line`/`_seqm_triton_source`/
  `_seqm_kernel_for`/`_seqm_binding`, `_SEQM_LOADED`,
  `scan_triton_bindings` seqm branch, `_site_parts` axes merge.
- `tests/test_discovery_play.py` — `TestSeqmSites` (mint,
  fold==chain plain+gated, handler axes, two declines).
- `tests/test_discovery_genkernel.py` — `TestSeqmSpec`,
  `TestSeqmDenotation`, `TestSeqmBindings` (fp64 CUDA kernel vs
  fold, gated variant, shape/cuda declines, on-disk source).
