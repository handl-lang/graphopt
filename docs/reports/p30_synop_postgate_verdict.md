# P30 — T-016 closure: post-all-gates synop delivery verdict

**Question (ledger T-016).** After the fused-body codegen path (p26),
the strided-kernel fix (p29), the fold cost gate (p27/p28) and the
claim cost gate (T-009/T-010), do **any** `synop*`/`ksynop*` folds
deliver a *verified wall-clock win* on the zoo corpus?

**Answer: no — and the boundary is now measured on both sides.**
Delivery is real (verified `synop_*` members ship through the fused
path on three rows, both dtypes); the pay is corpus-bounded: every
zoo prim site is a small elementwise fold where launch/host overhead
dominates, and the delivered *module* still loses to spelled eager —
and loses 3–7× to `torch.compile`. Machine-readable rows:
`zoo_chain_sweep_t016_postgate.json`.

## Method

Two passes over all 68 zoo workloads at the sweep shape
(`d64 t32 b8`, CUDA, RTX 2050), each dtype:

1. **Extraction scan** (no module timing): `_zoo_case` →
   `_gen_board` (runs `_claim_cost_gate` + `_fold_cost_gate`) →
   `_probe_play` → `deliverable` term → `prim_ops` = term ops ∩
   `SYNOP_SHIPPED`.  Identifies which rows can deliver prims
   post-gate.  ~5 s/row typical; the three recurrence modules cost
   60–1065 s each (saturation/extraction, not timing).
2. **Timed `gen_probe`** on every row the scan flagged, `reps=100`,
   serial (GPU timing serialized), repeated rounds on the heavy
   rows.  `speedup` = spelled-eager / gen; `vs_compiled` =
   compiled / gen (both arms timed inside `gen_probe`).

## Which rows deliver synops post-gate

| dtype | row | prim_ops in delivered term | timed runs |
|---|---|---|---|
| fp64 | GatedDeltaRule | `synop_4f8fefdd38` (run 1); `synop_39f76b66f8` (scan) | 1 |
| fp64 | GatedLinearAttention | `synop_2b8cfd1c90`, `synop_62d36aefdb`… (3 ops, varies) | 1 |
| fp64 | MambaSelectiveScan | 4 ops run 1 (`synop2_35bbd03301`, `synop3_0ca31bf1e5`, `synop3_0fa19d1ff8`, `synop_5e58f26875`); `synop2_35bbd03301` run 2 | 2 |
| fp32 | GatedDeltaRule | none (run 1); `synop_5147826430` (scan) | 1 |
| fp32 | GatedLinearAttention | `synop_2b8cfd1c90`,`synop_62d36aefdb`; `synop_571fb768b0`,`synop_5e58f26875`; `synop3_39dbcda988` | 3 |
| fp32 | MambaSelectiveScan | `synop_5e58f26875`; `synop_2b8cfd1c90`; `synop_f3290a1163` | 3 |

Which specific synop wins extraction **varies run to run** — the
fold gate's per-site referee has documented margin noise (T-026
residual), so borderline members flip between processes.  The row
set is stable: exactly these three recurrence-heavy modules carry
sitable, gate-surviving prim folds; the other 65 rows extract zero
prim ops in both dtypes.

## Timed results — every verified, every loses

`speedup` vs spelled eager; `vs_compiled` = torch.compile / gen.
All rows `verified=True` (`max_abs` ≤ 1.9e-6 fp32, ≤ 8.9e-16 fp64).

| row | dtype | run | prim_ops | speedup | vs_compiled | gen µs | eager µs |
|---|---|---|---|---|---|---|---|
| GatedDeltaRule | fp64 | 1 | 1 | **0.668** | 0.747 | 571 | 382 |
| GatedLinearAttention | fp64 | 1 | 3 | **0.596** | 0.371 | 3749 | 2236 |
| MambaSelectiveScan | fp64 | 1 | 4 | **0.481** | 0.131 | 4139 | 1990 |
| MambaSelectiveScan | fp64 | 2 | 1 | **0.517** | 0.142 | 3553 | 1838 |
| GatedDeltaRule | fp32 | 1 | 0 | 0.896 | 0.377 | 428 | 383 |
| GatedLinearAttention | fp32 | 1 | 2 | **0.629** | 0.162 | 3344 | 2104 |
| GatedLinearAttention | fp32 | 2 | 2 | **0.604** | 0.169 | 3148 | 1902 |
| GatedLinearAttention | fp32 | 3 | 1 | **0.587** | 0.161 | 3239 | 1900 |
| MambaSelectiveScan | fp32 | 1 | 2 | ~~2.193~~ artifact | 0.142 | 3940 | **8640** |
| MambaSelectiveScan | fp32 | 2 | 1 | **0.504** | 0.131 | 3811 | 1919 |
| MambaSelectiveScan | fp32 | 3 | 1 | **0.497** | 0.132 | 3675 | 1825 |

**The 2.193 draw is a contaminated arm, not a win.**  Its eager
baseline read 8,640 µs — 4.5× the stable ~1,800–2,100 µs the same
row measures in the other five runs (gen side was normal at
~3.7–3.9 ms).  Two clean re-runs of the identical row read
0.504 / 0.497.  Recorded in the artifact for honesty; excluded
from the verdict.

## The boundary, stated with numbers

Delivery is real; pay is corpus-bounded on **two** measured
boundaries:

1. **Per-site (p29, unchanged):** an elementwise prim fold's fused
   Triton body beats its spelled site iff the output is
   ≳ 0.5 M elements fp64 (1.10× @ 0.52 M, 1.46× @ 2.1 M, post
   strided-kernel fix).  Every zoo prim site is 256–32 k output
   elements — launch + host overhead (~25–45 µs/call) exceeds the
   ~7–9 µs/op spelled body, so no elementwise fold *can* pay on
   this corpus.  `_fold_cost_gate` correctly refuses the losers;
   the few members that survive do so on referee margin noise,
   and they still lose at module level (table above).

2. **Module level (this sweep):** on the only three rows that
   deliver prims, the delivered module runs 0.48–0.90× spelled
   eager (median ≈ 0.58 over the 11 clean timed runs) and
   0.13–0.38× `torch.compile`.  The recurrence modules' wall time
   is dominated by non-prim structure (the scan/carrier bodies and
   surrounding attention), so even a hypothetically free fold
   could not move the module ratio across 1.0 — the spelled eager
   baseline itself is only ~2 ms while `torch.compile` reaches
   ~0.5 ms.

**Verdict: T-016 closes as "delivery real, pay bounded at the
corpus's site sizes."**  The fused-body codegen path the item asked
for exists, verifies (≤1.9e-6), and is gated by measured
wall-clock; the remaining gap to a paying zoo row is a *site-size*
gap, not a machinery gap.  A corpus with ≥0.5 M-element elementwise
sites (or seq-scan/attr-bearing bodies once their residuals land)
is where the next pay would show.

## Residuals checked (both still open, unchanged)

* **Seq-form scan claims offer unpriced** — confirmed:
  `_claim_cost_gate.measurable` still excludes
  `gk._seq_scan_spec(pattern)` handlers (play.py ~line 866), and
  `hybrid_sink` leaves declined seq specs unbound rather than
  delivering the wrong denotation.  Needs a carrier-aware spelled
  referee; not touched here.
* **Attr-bearing bodies codegen contract** — confirmed:
  `triton_bindings` only models the elementwise grammar, so the 5
  attr-metavar prims still deliver via the `torch.compile`d
  spelled body (~0.4–0.7× per p29 §1).  Needs unrolled-op codegen;
  not touched here.

## Caveats

* Timed rows are single `gen_probe` calls (`reps=100`); run-to-run
  extraction noise means the *set* of delivered synops varies, but
  every run on every prim-bearing row lost — 11 clean timed runs,
  zero wins.
* The fold gate's margin noise let `synop_4f8fefdd38` deliver on
  GatedDeltaRule fp64 at 0.668× — a known-loser slipping through,
  the T-026 residual already on record; it does not change the
  verdict (it loses).
