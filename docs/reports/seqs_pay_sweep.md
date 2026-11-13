# seqs pay sweep — the measured boundary curve

Date: 2026-11-14.  Follow-up to `docs/reports/cumsum_carrier.md`,
which minted the `kind="seqs"` emitted-scan tier, verified it
bit-exact, and recorded an honest near-parity verdict on one shape
(`(8, 32, 64, 64)` fp64 CUDA: kernel ~205–230 µs vs spelled
`torch.cumsum` ~200–211 µs, both referee windows keep-vote False).
This report sweeps the shape space to answer: **is there ANY
realistic regime where the emitted Triton scan beats the cub-class
incumbent?**

Probe: `bench/drivers/probes/seqs_pay_sweep.py` (standalone,
serial GPU, `gk._paired_time` interleaved windows — the same
primitive `profitable` uses).  Host: RTX 2050 4 GB, torch
2.14.0+cu130, triton 3.8.0.  fp64 throughout unless marked fp32
(the intake convention; fp32 spot-checked at the end).

## Verdict

**No stable paying regime found.**  The kernel never wins by more
than noise anywhere in the swept space, the in-module referee
declines the claim at every zoo shape tried — including the one
standalone shape that transiently read sub-1.0 — and `T > 256`
cannot run at all (`_SEQS_MAX_T` tile cap → `ValueError` → the
referee declines).  cumsum is **covered-but-not-paying**, now with
the boundary curve measured rather than assumed.

## The strided `_gs` boundary curve — `(B, T, N)` axis=1, B=8, fp64

`spelled` = `torch.cumsum(x, dim=1)` on the same CUDA tensor;
`kernel` = the generated binding (verify-before-time passed at
every point, `verify=ok`).  One process, `reps=30`, interleaved:

| T \\ N | 64 | 256 | 1024 |
|---|---|---|---|
| 8   | 2.05 | 1.99 | 1.95 |
| 16  | 1.90 | 1.69 | 1.21 |
| 32  | 1.38 | 1.31 | 1.11 |
| 64  | 1.14 | **0.90** | 1.16 |
| 128 | 2.21 | 2.50 | 3.83 |
| 256 | 2.22 | 2.42 | 3.71 |

(kernel/spelled ratio; <1.0 = kernel faster.  Absolute times:
spelled 9–400 µs, kernel 19–1470 µs across the grid.)

The shape is a valley, not a plateau: the kernel has a ~19–24 µs
floor (one launch + tile setup) that loses to `cumsum`'s ~9–15 µs
at small sizes, approaches parity around `T≈32–64, K≈256–1024`,
then **collapses past `T=64`** — `BT = next_pow2(T)` hits 128/256
and the `BT × BN` tile spills/serialises (ratio 2.2–3.8×).  The
`tl.cumsum`-inside-one-tile design has no answer for `T > 256`:
the cap is honest, not a tuning miss.

### The `(8, 64, 256)` "island" is noise — measured, not assumed

That cell read **0.897** in the consolidated run above — the only
sub-1.0 in the whole grid.  It does not survive scrutiny:

- Five consecutive `_paired_time` windows in one process read
  0.921–0.953, then a *later* window in a *later* process read
  1.9, then the consolidated re-run read 0.897 and its B-sweep
  neighbour `(8, 64, 256)` read **1.049** minutes later in the
  *same* process.  The spelled arm itself swings 40–65 µs across
  sessions (GPU at 73 °C, SM clock 1327 MHz — this host's thermals
  make ±30% drift ordinary; the T-026 residual the referee's
  borderline band exists for).
- The decisive check is the referee itself: `play._zoo_case` on
  `CumsumLinearAttention` at `zoo(d=16, t=64, b=8)` — whose seqs
  operand is exactly `(8, 64, 16, 16)`, `K = d² = 256`, the island
  shape — timed `spelled=66 µs, kernel=107 µs`, **keep=False**.
  Every in-module row below declined.

So: a ~5–10% transient edge in a noise band is not a regime.  The
referee's majority-vote-over-windows is doing exactly its job.

## Zoo-realistic 4-D operand — `(B, T, d, d)` axis=1, B=8, fp64

This is the shape `CumsumLinearAttention` actually mints
(`kv.cumsum(1)`, `K = d²`).  `d=64` reproduces the report's
operand family:

| T | spelled µs | kernel µs | ratio |
|---|---|---|---|
| 8   | 56.1  | 64.0   | 1.14 |
| 32  | 199.7 | 204.3  | **1.02** ← the report's shape |
| 64  | 384.3 | 402.9  | 1.05 |
| 128 | 769.6 | 3180.6 | 4.13 |
| 256 | 1548.4| 6195.0 | 4.00 |

Near-parity only in the `T ≤ 64` band where `K = 4096` gives the
tile enough columns to amortise setup; past `T=64` the `BT`
explosion dominates and the kernel is 4× worse.  Smaller `d`
(equivalent `K` 64–1024 in the 3-D table) behaves identically —
`K` is what matters, not the rank.

## `T > 256` — cannot run, by construction

```
(8, 512, 256)    DECLINE ValueError: seqs scan step extent 512 > 256   spelled  282 µs
(8, 1024, 256)   DECLINE ValueError: ... 1024 > 256                    spelled  544 µs
(8, 4096, 256)   DECLINE ValueError: ... 4096 > 256                    spelled 2174 µs
(8, 16384, 256)  DECLINE ValueError: ... 16384 > 256                   spelled 7827 µs
(8, 1024, 16,16) DECLINE (4-D)                                         spelled  498 µs
(8, 4096, 16,16) DECLINE (4-D)                                         spelled 2005 µs
```

The incumbent keeps scaling (cub handles any `T`); the emitted
kernel has no multi-tile/decoupled-lookback path, so the whole
`T > 256` half-plane is unreachable — the largest honest gap
between the two arms.

## The dense `_gsc` fallback — always loses

Non-contiguous operand (permute-forced `(8, T, N)`), fp64:

| shape | spelled µs | kernel µs | ratio |
|---|---|---|---|
| (8, 64, 256)   | 65.4  | 58.0   | 0.89* |
| (8, 256, 256)  | 245.9 | 487.6  | 1.98 |
| (8, 64, 1024)  | 201.8 | 271.2  | 1.34 |
| (8, 256, 1024) | 770.3 | 2435.3 | 3.16 |

\* the one sub-1.0 is the same noise band as above (adjacent
windows read >1); the structural story is the ~`T·N` transpose
copy the dense path pays up front.  Multi-dim prefixes
(`(B,H,T,d)` axis 2) always land here — `s > 1` fails
`_seqs_strided_view` — so headed-scan spellings can never reach
the strided kernel at all.

## Fold mode and dtype edges

- `"fold"` (generic emitted recurrence, any `T`): `(8,256,256)`
  ratio 3.11, `(8,1024,64)` ratio 2.33 vs `cumsum` — the serial
  step loop is strictly worse, as documented.
- **fp32:** same losing profile ((8,64,256) 1.28, (8,64,1024)
  1.18) plus a new honest edge: `(8,128,256)` fp32 **raises**
  `OutOfResources` (required 131 KB > 101 KB shared limit) where
  fp64 at the same tile runs.  Shared-memory pressure is
  dtype/config-dependent and the binding does not preflight it —
  the referee's verify-before-time catch turns it into a decline,
  which is correct but worth recording: "dtype must match" is not
  the only raise condition.
- **`_SEQS_BN` sweep:** `BN=512` is uniformly worse (3.8× at
  (8,256,1024) and (8,64,4096)); `BN=128` is mixed — one
  `OutOfResources`, one 1.08.  `BN=256` is not obviously wrong,
  just not sufficient.

## In-module referee — the authoritative verdict

`play._zoo_case` on `CumsumLinearAttention`, six `(d, t, b)`
points, `gk.PROFITABLE_LOG` rows for `scan_0` (spelled µs /
kernel µs / keep-vote):

| (d, t, b) | operand | spelled | kernel | votes |
|---|---|---|---|---|
| (64, 32, 8)  | (8,32,64,64)   | 251.8 / 249.8 | 310.8 / 307.7 | F, F |
| (16, 64, 8)  | (8,64,16,16)   | 66.1           | 107.1         | F |
| (16, 32, 8)  | (8,32,16,16)   | 52.1           | 108.0         | F |
| (16, 128, 8) | (8,128,16,16)  | 111.5          | 278.9         | F |
| (64, 256, 8) | (8,256,64,64)  | 1682.2         | 6400.6        | F |
| (128, 64, 8) | (8,64,128,128) | 1670.7 / 1675.8| 1808.1 / 1889.0| F, F |

`gen_handlers == {}` at every point — the offer is empty, exactly
the honest-decline contract.  `TaylorLinearAttention` /
`ChunkedSSD` mint the same site shape on smaller tensors and are
likewise covered by the curve above (same `K`, smaller `T` → the
small-T losing band).

## Does the machinery need shape routing? (task item 4)

**No routing gap for the regime that exists — and none can help
the regime that doesn't.**  The binding already picks strided vs
dense per call from the operand's strides (`_seqs_strided_view`),
and the referee times on the *real bound values*, so any shape
that genuinely paid would be kept automatically — no spec/shape
plumbing is missing for delivery.  The actual gaps, precisely:

1. **`T ≤ _SEQS_MAX_T = 256` hard cap** — half the requested grid
   (`T ∈ {512…16384}`) is unreachable by construction; beating
   cub there needs the decoupled-lookback multi-kernel scan the
   carrier report already scoped.
2. **`BT = next_pow2(T)` tile quantisation** — `T=48` and `T=64`
   compile the same `BT=64` tile but the masked waste shows up
   directly in the ratio; there is no sub-tile `T` handling.
3. **No shared-memory preflight** — `OutOfResources` raises are
   only caught downstream (referee decline / would-be runtime
   error); a `BT × BN × elt_size` check in `_seqs_binding`
   would make the decline explicit and earlier.
4. **Prefix rank ≤ 1 for the strided path** — headed/multi-dim
   prefixes always pay the dense transpose; a `(P1, P2, T, K)`
   4-D strided launch is the obvious extension if a paying dense
   regime ever appears (it doesn't — dense loses everywhere).

## Bottom line for the corpus

`cumsum`/`cumprod` carriers are **minted, denoted, bound,
verified — and measured not to pay** against `torch.cumsum`'s
cub scan on every shape probed: small `T` loses on launch/setup
overhead (~19–24 µs floor vs ~9–15 µs), `T ≈ 32–64` with large
`K` reaches parity (1.02–1.16, noise-crossing), `T ≥ 128` loses
2–4× on tile spill, `T > 256` cannot execute, and non-contiguous
or headed operands pay a transpose that loses outright.  The one
transient sub-1.0 reading was shown to be host noise, not a
regime.  Coverage claim stands; pay claim does not — and now the
boundary is a curve, not a single point.
