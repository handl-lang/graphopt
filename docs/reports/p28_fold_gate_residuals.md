# P28 — T-026 fold-gate residuals: attr-metavar siting, referee device, measured rms_norm

Follow-up to `p27_fold_cost_gate.md`. Three residuals filed against the
T-019 gate, plus the two law-fold misses the ledger cited. Machine-readable
rows: `zoo_chain_sweep_t026_fp64.json` (7 prim rows + `RMSNormPerHead`,
fp64, 3 rounds, isolated) and `zoo_chain_sweep_t026_fp32.json`
(`SigmoidAttention`, fp32, 3 rounds, isolated).

## 1. Attr-metavar prim specs — sited, not dropped

5 of the 57 shipped prim denotation specs carry attr metavariables
(`{'dim': 'a97'}`-style names inside attr dicts). `_concrete_match`
compares attrs literally, so they could never site and kept their
`supported` seat unconditionally — `GatedDeltaRule` fp64 delivered
`synop_4c5296b3ba` + `synop_5e58f26875` at 0.54×.

**Fix.** `genkernel.attr_site_env` / `concretize_at_site`: a matcher
that binds attr-mvar *names* from the matched site's real attrs
(consistent binding, literal compare otherwise, same strictness as
`_match_node`), then freezes them via `spec_bind` into a concrete
spec copy. `play._fold_cost_gate` sends that copy through
`profitable`; the **delivered** table keeps the polymorphic original
(one site's frozen attrs would mis-denote another site's call).

**But siting alone didn't fix GatedDeltaRule** — the frozen spec still
won the referee (0.956→ kept at first, module still 0.535×). The real
bug was underneath:

## 2. The referee measured the wrong device

For handlers every codegen tier declines (all 5 attr-metavar specs —
`unsqueeze`/`select`/`linear`/`matmul` bodies), `kernel_for` returns
`None` and `_referee_kernel` fell back to `torch.compile`d spelled
eval **on the CPU env** — while `hybrid_sink` delivers onto CUDA.
Measured directly on the `synop_4c5296b3ba` site (fp64, CUDA):

* compiled spec-walk binding: **55.8 µs**
* uncompiled spec-walk binding: 39.7 µs
* eager `_eval_spec`: 34.9 µs

`torch.compile` graph-breaks on the recursive `_eval_spec`/`spec_bind`
walk and pays per-call dynamo guard overhead the eager spelled side
never pays — on CUDA. On CPU the same comparison read a win, which is
exactly the "0.64× vs ~2.2× across processes" slip the ledger cited
(the 0.64 draw was the CPU-fallback artifact landing on a borderline
spec; `SYNOP_SHIPPED` being a `frozenset` also made *which* handler
absorbed first-call warmup noise a per-process coin flip).

**Fixes, all in `genkernel`:**

* `_referee_kernel` fallback now moves the env to CUDA when available
  (move failure keeps the host env) — the referee times what delivery
  ships, on the device it ships to.
* `_paired_time` replaces the two sequential `_median_time` windows
  in `_kernel_pays`: both sides warm up together, then alternate
  a-then-b / b-then-a inside one loop, so slow drift cancels instead
  of landing wholly inside one arm.
* `profitable` iterates `sorted(handlers)` — deterministic order
  across processes.

Re-check of `synop2_35bbd03301` across fresh processes: **2.04–2.21**
(five reps each, was 0.64 vs ~2.2). It now declines consistently —
correctly: the generated Triton kernel is ~2× slower than spelled at
that site.

## 3. Law folds — `rms_norm` gets a measured delivery dispatch

`rms_norm_fold`/`_nogain` are bank laws in `DEFAULT`: they fire in
saturation and extraction prices the `rms_norm` member at one op —
a cost-model win no referee ever sees (profitable only gates handler
*claims*; law folds rewrite to ambient ops and never pass through it).
On this box `F.rms_norm`'s fp64 kernel is pathological at
head-width shapes: **222 µs vs 30 µs** spelled (fp32: 22.5 vs 30 —
fused wins). `RMSNormPerHead` fp64 delivered `rms_norm` and read
0.15–0.21× while verifying perfectly.

A site-level gate cannot apply: the fold fires inside saturation on
the e-graph, the choice happens at extraction under a static cost
model, and `cost_fn` sees IR nodes, not bound values — there is no
place to insert a wall-clock measurement without timing every
candidate member per extraction (prohibitively expensive, and
nondeterministic).

**What shipped instead** — the equivalent gate one level down, where
the measurement *can* live: `genkernel._rms_norm_measured`, installed
via `_ambient_bindings()`/`_MEASURED_BINDINGS` into all five generated
-kernel sinks (`gen_sink`, `hybrid_sink`, `triton_sink`,
`scan_triton_sink`, `decayed_triton_sink`). The op's denotation is
dtype/shape-polymorphic; its implementation is not — the binding
times `F.rms_norm` vs the fold's own LHS spelling once per
`(dtype, device, normalized_shape, has_weight, shape)` and caches the
winner (bounded 64-entry cache). Same function either way — verify
is unaffected; the law fold keeps its cost win where the fused kernel
pays and silently delivers the spelled body where it doesn't.

## 4. `SigmoidAttention` fp32 — not a fold miss

The delivered term is **structurally identical** to the source IR
(34 nodes, op-for-op, attrs included — diffed directly). No law fold
survives extraction; `prim_ops: []`. The 0.72–0.79× is the
binding-dispatch floor: the lowered module pays per-op Python
dispatch through `OpTable` bindings where the eager `nn.Module`
makes direct calls — ~40–60 µs on a ~160 µs launch-bound module
(profiler shows identical CUDA kernel counts/times on both sides).
No `profitable`-style gate can see it because no fold fired; the fix
would be compiled whole-module delivery, which is a different
mechanism. Documented, still open — re-measured 0.794× (3 rounds,
spread 0.726–0.799).

## Re-sweep (post-fix, fp64 unless noted; median of 3 isolated rounds)

| case | p27 gated | T-026 | prim ops delivered |
|---|---|---|---|
| GatedDeltaRule | 0.542 | **0.956** | — (was `synop_4c5296b3ba`) |
| DecoupledMLA | 0.980 | 0.955 | — |
| DilatedBlockAttention | 0.833 | 0.852 | — |
| GRUFusedGate | 0.770 | 0.932 | — |
| ChunkedSSD | 0.972 | 1.020 | — |
| RWKVMixBlock | 1.114 | 0.988 | — |
| HighwayGate | 0.940 | 0.953 | — |
| **median of 7** | **0.940** | **0.955** | |
| RMSNormPerHead (fp64) | 0.151–0.21 | **0.876** | — |
| SigmoidAttention (fp32) | 0.72 | 0.794 | — |

The 7-row prim median 0.940 → 0.955 and `GatedDeltaRule` went from
the worst row (0.542, delivering two losing synops) to median
(0.956, delivering none). `RWKVMixBlock`'s 1.114 → 0.988 is claim-
admission noise (no prim ops either run), same class p27 documented.
`RMSNormPerHead` 0.21 → 0.876: the measured dispatch picked the
spelled body on fp64 as designed; the residual <1 is the same
dispatch-overhead floor as `SigmoidAttention` on a 35–45 µs module.

## Verdict

- **Attr-metavar gating: fixed.** The 5 specs now site through
  `concretize_at_site`; `synop_4c5296b3ba` is measured and refused.
- **Referee noise: fixed at two roots.** Sequential timing windows →
  interleaved paired timing; CPU-fallback → delivery-device (CUDA)
  measurement; frozenset iteration → sorted. `synop2_35bbd03301` is
  stable at ~2.1× slower across processes and declines consistently.
- **Law-fold gate: partial, honest.** `rms_norm` delivers through a
  measured per-site dispatch (0.21 → 0.88). A fold-level measured
  gate cannot exist — saturation/extraction never sees values; the
  dispatch is the closest honest equivalent. `SigmoidAttention` is
  not a fold issue at all — delivery dispatch overhead, documented.
