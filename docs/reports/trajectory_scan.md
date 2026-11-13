# Trajectory / read-out scan tier — scoping and prototype

**Date:** 2026-11-13 · **Builds on:** `module_loss_gla_mamba.md` (the
seq/seqm route delivers+verifies on GLA/Mamba but loses at module
level because the modules return `torch.stack(outs)` — every step's
read-out — while the shipped kernels emit only the final state).

**Short answer: the trajectory half of the tier ALREADY EXISTS** —
`kind="seqs"` specs denote and kernel-emit every prefix state
(`out_t = acc_t`), with an independently spelled denotation
(`genkernel.seqs_denotation`).  What does not exist is the
*read-out*: GLA/Mamba need `o_t = contraction(S_t, per-step operand)`,
not `S_t` itself.  Prototyped the honest middle path — **seqs
trajectory kernel + ONE batched einsum for all T read-outs** — with
zero new kernel code, through the real `scan_spec` /
`scan_triton_bindings` / `seqs_denotation` machinery.  It verifies
(`allclose(1e-5)`, max_abs ≤ 4.6e-5 vs the spelled fold) and pays:
**2.7× (GLA) / 14.5× (Mamba)** over the spelled recurrence+read-out
on identical operands, projecting to ~**2.7× / ~9× vs the eager
module** and ~**6× / ~9× vs today's delivered module** — *if* the
collector/route/read-out-spec machinery existed.  That machinery is
the residual; it is scoped in §5.

## 1. What the scan tiers already express

`genkernel.scan_spec` recognises three carrier kinds
(`genkernel.py:1740-1877`):

| kind | denotes | kernel | minted from |
|---|---|---|---|
| `seq` | final state `h_T` of `acc = a_t·acc + b_t` | `_gs` serial fold (`_seq_triton_source`, :2353) | unrolled `add(mul(·,·),·)` chains via `_seq_scan_sites` (`play.py:1883`) |
| `seqs` | **every prefix state** `out_t = acc_t` | `_gs` serial fold with per-step store (`_seqs_fold_source`, :2688), or `tl.cumsum`/`tl.cumprod` tiles for closed forms (`_seqs_scan_source`, :2626) | `cumsum`/`cumprod` ops only, via `_cumsum_sites` (`play.py:2110`) |
| `seqm` | final state of the rank-1-feedback matrix fold | `_gsm` tiled kernel (`_seqm_triton_source`, :3215) | state-feedback chains via `_seqm_scan_sites` (`play.py:2050`) |

The `seqs` denotation (`seqs_denotation`, :2003) spells the same
recurrence with `outs.append(acc)` per step and `torch.stack` back
along the step axis — trajectory output is *already* an independently
checkable denotation, exactly the property the task asked for.  Its
kernel is the `seq` kernel plus one `tl.store` inside the step loop
(:2724).  Constraint: minted only where a `cumsum`/`cumprod` op
exists — no collector yet mints `seqs` from a recurrence chain whose
intermediate states are *consumed* (the GLA/Mamba shape).

## 2. What the workloads actually need

Both losers return per-step read-outs that are **contractions of the
post-update state along one state axis** with a per-step operand:

- **GLA** (`zoo.py:812-816`): `S ← diag(g_t)·S + k_t v_tᵀ`,
  `o_t = einsum("bdv,bd->bv", S_t, q_t)` — contracts state axis 1
  (`d`, size 64); output `(B,T,d)`.
- **Mamba** (`zoo.py:772-777`): `h ← dec_t⊙h + x_t B_tᵀ`,
  `o_t = einsum("bdn,bn->bd", h_t, c_t)` — contracts state axis 2
  (`n`, size 4); output `(B,T,d)`.

Neither needs `h_t`/`S_t` itself — only a rank-`(state-1)` projection.
Two honest designs follow:

- **(a) trajectory + fused read-out** *(prototyped)*: `seqs` kernel
  emits `(T, *state)`; ONE `einsum` contracts the trajectory with the
  stacked per-step read-out operand — `"btij,bti->btj"` /
  `"btdn,btn->btd"` — replacing T separate `select`+`einsum` pairs.
  Works with shipped machinery today; costs a `(T,·,state)` write
  (4 MB for GLA) plus the einsum's permute.
- **(b) fused `seqt` kernel** *(designed, not built)*: the fold
  carries state AND writes only `o_t` per step — programs own a
  `(outer…, contract-C)` state slice so the read-out contraction is
  a local `tl.sum` (the same trick `_gsm` uses for `Kᵀ·acc`,
  :3232-3235).  Writes `(T,·,out)` instead of `(T,·,state)` — for
  GLA ~256 KB vs 4 MB, and no einsum at all.  This is the real end
  state; §5 scopes it.

## 3. Prototype — spec, denotation, kernel, verification

Specs built by hand in the collector's own format (the same shapes
`_seq_deindex`'s view-lift produces — `unsqueeze`d sequence operands,
bare-meta `a`, elementwise `b`):

```python
# GLA — state (B,d,d); a_t = gg_t (B,T,d,1); b_t = kk_t·vv_tᵀ
spec = ("applyd", ("aff_diag", "G", ("mul", "K", "V")), "H",
        {"kind": "seqs", "axis": 1})
traj = binding(gg.unsqueeze(-1), kk.unsqueeze(-1),
               vv.unsqueeze(-2), zeros(B,d,d))   # (B,T,d,d)
out  = einsum("btij,bti->btj", traj, qq)          # (B,T,d)

# Mamba — state (B,d,n); a_t = dec_t; b_t = x_t·B_tᵀ
spec = ("applyd", ("aff_diag", "D", ("mul", "X", "B")), "H",
        {"kind": "seqs", "axis": 1})
traj = binding(dec, x.unsqueeze(-1), bb.unsqueeze(-2),
               zeros(B,d,n))                      # (B,T,d,n)
out  = einsum("btdn,btn->btd", traj, cc)          # (B,T,d)
```

`scan_spec` accepts both (`kind="seqs"`, `scan_mode="fold"` — the
generic serial emitted fold, any `T`); `scan_triton_bindings` emits
the `_gs` store-per-step kernel through the normal cache path.

**Verification** (zoo(64,32,8) shapes, fp32 CUDA, `torch.manual_seed(0)`):

| check | GLA | Mamba |
|---|---|---|
| traj+einsum vs `module(x)` | max_abs **2.9e-6**, `allclose(1e-5)` ✓ | max_abs **7.2e-7**, `allclose(1e-5)` ✓ |
| spelled loop vs `module(x)` | **0.0** (identical op order) | **0.0** |
| `seqs_denotation` vs kernel traj | max_abs 4.6e-5, `allclose(1e-5)` ✓ | — |

The residual ~1e-6–5e-5 is fp32 accumulation-order noise: the Triton
fold fuses `a*acc+b` into one FMA-rounded step where eager rounds
twice, and the batched einsum reorders the contraction vs the T
separate einsums.  Same class of delta every generated tier already
verifies under.

## 4. Measured cost — does it pay?

fp32, same precomputed operands (projections excluded — they are
shared with every arm), synced wall µs, reps=50, RTX 2050:

| arm | GLA | Mamba |
|---|---|---|
| spelled recurrence + T read-outs (eager-equivalent) | **1277** | **1255** |
| `seqs` kernel + broadcast rows | 337 (device 334) | 63 (device 36) |
| fused read-out einsum | 142 (device 142) | 30 (device 12) |
| **trajectory path total** | **479** | **~93** |
| speedup vs spelled | **2.7×** | **14.5×** |
| + projections (~38 / ~32 µs) → vs eager module (~1400 / ~1390) | **~2.7×** | **~9×** |
| vs *today's delivered* module (report §2: 3208 / 3502 µs wall) | **~6×** | **~9×** (projected — see §5) |

Component honesty for GLA's 337 µs kernel-side: ~**151 µs is the
`_seqs_fold_binding` broadcast materialisation** — three
`(B,T,d,d)`-broadcast operands contiguous'd to `(T,N)` rows (12 MB)
— the same `_seq_args` weakness the report flagged for `_gs`
(:2509-2550), plus a 4 MB trajectory write inside `_gs` and ~186 µs
of actual fold kernel.  A stride-aware (`ometa`-style) seqs variant
would recover most of the 151 µs; design (b) would also erase the
4 MB write and the 142 µs einsum.

## 5. What is missing for production — the residual

The prototype proves the *kernel semantics*; it is not yet
*deliverable*.  The gap is machinery, not math:

1. **A `seqs`-from-chain collector.**  `_cumsum_sites` is the only
   `seqs` source; GLA/Mamba need a `_seq_scan_sites`-style walk that
   mints `kind="seqs"` when the chain's intermediate states have
   *consumers* (the per-step read-outs).  The site record must also
   capture the read-out structure — today nothing records "these T
   einsums share one operand sequence and contract one state axis".
2. **A read-out spec + route.**  `_route_scan_chains`
   (`meta_arena.py:1083`) replaces the chain *root* (final state)
   only.  Trajectory delivery must additionally rewrite each
   intermediate-state consumer — either `select(traj, t)` feeding
   the spelled einsums (works today, keeps 32 einsums) or a new
   minted read-out op / the fused `seqt` kernel of §2(b).  The
   fused single einsum is itself a new term shape the sink must
   lower — or the `seqt` kind carries `"readout"` in its spec attrs
   and `seqt_denotation` spells fold+contract (still independently
   checkable against the spelled chain).
3. **`seqt` kernel (design b).**  Program per `(outer, contract-block)`
   state slice; per step: elementwise `acc = a_t·acc + b_t` (the
   existing expr renderer), then `o_t = tl.sum(acc · w_t)` over the
   contract axis, stored to `(T, out…)`.  Operand role resolution
   mirrors `_seqm_signature` (:3064) — each operand's rest dims are
   classified against the `(outer, C)` state layout at call time.
   Estimated headroom over the prototype: erase ~151 µs broadcast
   copies (strides), ~4 MB traj write, and ~142 µs einsum → GLA
   floor ≈ **250–350 µs** device vs the prototype's ~476.
4. **Referee wiring.**  `profitable`'s `denotation_for` already
   handles `seq`/`seqm`/`seqs` via `handler_denotation`
   (:2324-2350); a `seqt` kind slots in identically — the spelled
   arm is the fold+readout chain, the priced arm the kernel.

**Pay verdict: YES at kernel level, projected YES at module level —
contingent on the §5 machinery.**  Unlike the final-state route,
which *added* a scan on top of an un-foldable spelled recurrence,
the trajectory path *replaces* the whole recurrence+read-out body:
the spelled loop it beats is the same 32-step chain eager runs, and
the win is structural (one launch + one einsum vs ~200 small ops),
not a per-site margin.  The residual loss versus a hypothetical
perfect delivery is the broadcast materialisation and the
trajectory write — both bounded and both removable by (b).

## Caveats

- Prototype operands were computed by replicating each module's
  projection prefix by hand (verified: the spelled-loop arm is
  bit-exact vs `module(x)`, so the operand extraction is exact).
  Projection GEMMs are excluded from the recurrence arms and added
  back only in the module-level projection.
- Timing is wall-synced on an RTX 2050 (4 GB) — same class of
  measurement as `module_loss_gla_mamba.md`; absolute µs vary with
  mint/host but the 2.7×/14.5× recurrence-level ratios are
  structural.
- `seqs` fold mode is serial in `T` — fine at T=32; the
  `tl.cumsum` closed forms do not apply (these are gated recurrences,
  `scan_mode="fold"`).
- No package files were touched; nothing here is wired into
  `TIER_SITES`, `handler_denotation`, or the sinks yet.

## Files / commands

- Probe scripts (scratch, not committed): `/tmp/traj_probe.py`
  (spec construction, verification, timing), `/tmp/traj_parts.py`
  (kernel/einsum component split + profiler device times).
- Code read: `genkernel.py:1533-1586` (`_seq_scan_spec`/`gen_bindings`
  consult), :1740-1877 (`scan_spec`/`_scan_kind`), :1843-1857
  (`_seqs_mode`), :1923-2096 (`seq_scan_denotation`/`seqs_denotation`),
  :2324-2350 (`handler_denotation`), :2353-2392 (`_seq_triton_source`/
  `_gs`), :2613-2725 (`_seqs_*_source`, the store-per-step kernel),
  :2764-2920 (`_seqs_*_binding`), :2975-3368 (seqm tier — the
  `(outer, contract)` program-decomposition precedent for `seqt`),
  :3371-3421 (`scan_triton_bindings`), `play.py:1657-1780`
  (`_seq_deindex` view-lift), :1883-2175 (the three chain
  collectors), `meta_arena.py:1083-1139` (`_route_scan_chains`),
  `zoo.py:735-816` (the two module definitions).
