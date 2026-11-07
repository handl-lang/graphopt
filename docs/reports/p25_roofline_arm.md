# P25 — The arithmetic was the lever: 47 % → 82–86 % of roofline, landed

T-022 asked for `down_proj` from ~47 % toward 70 % of the memory roofline.
The arm reached **81.9 % (30.0 µs)** on `down_proj` and **86.3 % (28.5 µs)**
on `gate_proj` — Marlin's recorded range (86–90 %) — with cosine 1.000000
against the torch dequant denotation. The fix is landed in `compile-triton`'s
emitted body and the vendored kernels are regenerated.

## Caveat on operands (honest scope)

The Qwen AWQ checkpoint and vllm are **not installed on this host**, so
`awq_gemm` / Marlin could not be re-run. All numbers are on **synthetic
standard-order operands** (`qw` int32 `[K,N/8]`, `qz` `[K/128,N/8]`, `sc`
`[K/128,N]`, `x` `[K]`), gated at cosine ≥ 0.9999 (achieved: ≥ 0.9999998 on
seeds 0–3) against `unpack_ref` — the same denotation the IR term carries.
Percent-of-roofline uses the same 89.1 GB/s denominator as P21–P24, so the
% column is directly comparable; the ×-vs-incumbent column is not
re-measurable here and is reported as such. One instrument note: the
*emitted baseline itself* drifted between process runs (47.2–53.5 µs —
GPU clock states), matching P24's documented noise floor; within-run
comparisons are consistent, and the final table reports medians of
3 × (20-call graph × 5 replays).

## What worked, in order of pay

**1. Group-hoisted dequant — the big one (46 % → 66 %).**
The emitted body computed `(q − z)·s` per element, i.e. it unpacked `qz`
and gathered `sc` over the whole `[BLOCK_K, BLOCK_N]` tile — and since a
k-block sits inside one quant group, it loaded and unpacked the *same*
zero row 128 times. Algebraically
`Σ (q−z)·s·x = s·(Σ q·x) − s·z·(Σ x)`, so the z-unpack and s-gather drop to
O(BLOCK_N) per block and the inner loop keeps only the weight unpack, one
multiply, and the sum. 53.3 → 37.2 µs.

**2. fp16 magic-number unpack (66 % → ~78 %).**
`(w >> s & 0xF) | 0x6400` is `1024 + q` reinterpreted as fp16 — a shift and
one LOP3-fusable and/or, then an fp16 subtract of 1024 — replacing the
int→float convert. Products stay fp16 (HFMA2-rate), the k-sum accumulates
fp32. 37.2 → 31.6 µs.

**3. int64 loads + config (→ 82–86 %).**
`qw` read as `int64 [K, N/16]` halves the load instructions on the dominant
stream; `bn/bk/warps` re-swept around it. Best cells: `down_proj` split8
bn64/bk64/w2 → **30.09 µs (81.7 %)**; `gate_proj` split8 bn128/bk64/w4 →
**28.58 µs (86.0 %)**. Re-measured through the vendored modules: 30.02 /
28.47 µs, cos 1.000000.

## What did not work (measured, so nobody retries)

| Variant | Result | Why |
|---|---|---|
| fold the 1024 into the group correction (`magic_fold`) | **REJECTED, cos 0.9994** | the fp16 product `m·x` rounds at 2⁻¹¹ relative on a ~1024-weighted term that is then differenced — the predicted cancellation, measured |
| `tl.fma(m, x, −1024x)` | 36.3 µs, slower than `magic` | explicit `tl.fma` doesn't beat ptxas contracting `(m−1024)·x`; likely an fp32 path |
| `tl.dot` tensor cores | 60–66 µs | the [16,BK] broadcast + layout conversion costs more than HMMA saves at GEMV sizes |
| inline-asm `mov.b32`→2×fp16 pair extraction | 42 µs | the register interleave costs more than halving the unpack saves |
| fp16 `tl.sum` | 32–36 µs, cos 0.999999 | slower *and* thinner on the gate |
| plain `int→fp16` convert (`i2f`) | 29–33 µs | competitive on gate, loses to magic on down |
| `max_contiguous`/`multiple_of` hints | neutral | the qw row already vectorises |

## The `gate_proj` question

P24: our kernel ran 52.3 µs on *both* shapes while `awq_gemm` was relatively
stronger on the n-heavy one (hence 0.87× there, 3.73× on down). The arm's
read: the old body was arithmetic-bound with **shape-invariant work per
byte** — same unpack+reduce instructions per weight element regardless of
geometry — so it produced identical µs on both, and the apparent "gate
weakness" was incumbent-side shape tuning, not a defect in ours. The new
body is still shape-agnostic (28.5–30.0 µs on both), now at the roofline
bandwidth Marlin achieves; whether the ×-vs-incumbent flipped needs a
checkpoint-bearing host.

## Landed

- `compile-triton/src/codegen.rs` — `_stage1` is now the hoisted-magic body;
  `_stage1_w64` added; `run` picks w64 iff `qw` is contiguous with even
  word-rows, refuses `bk > GROUP` (the hoist's precondition). Signature
  `run(x, qw, qz, sc, n, k, bn, bk, warps)` unchanged; the 8 crate tests pass.
- `lib-triton-qgemv` — kernels regenerated (`qgemv_k4864_split8`,
  `qgemv_k896_split8`; `gate_proj` re-routed split4→split8), manifest rows
  updated with the synthetic-operand caveat, 3/3 binding tests pass.
- The emitted body needs `BLOCK_K ≤ GROUP` — true for every config the
  search used; the guard fails loudly rather than silently wrong.

Numbers: `docs/reports/p25_roofline_arm.json`. Driver:
`bench/drivers/probes/awq/roofline_arm.py` (standalone,
`PYTHONPATH=lib-kernel-ir/src:<dir>` + `COMPILE_TRITON_BIN`).
The P25 lowerer build lives at `/tmp/ct-p25/target/release/compile-triton`
(the crate cannot build in-tree — the workspace `.cargo` patch table;
build from a copy outside `handl-lang/`).
