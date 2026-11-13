# p36 — seq view-lift: `unsqueeze`/`squeeze` step exprs mint honest seq scans; two more zoo sites pay

Date: 2026-11-12.  Scope: `mlopt_discovery.play` (`_seq_view_lift` +
the `_seq_deindex` view branch), `tests/test_discovery_play.py`
(`TestSeqScanSites` +3), `mlopt_discovery.domain_spec` (`deliver`
spec field — the Stage-2 leftover), `domains/torch-silu.json`,
`tests/test_domain_spec.py`.  Motivation: p35's honest boundary —
"`unsqueeze`-bearing step exprs decline at `scan_spec` (no
elementwise render)".  GatedLinearAttention was the named declined
site; the same grammar gap also held MambaSelectiveScan.

## The lever — lift the view onto the sequence child, not into the step expr

The naive widening (teach the Triton step-expr renderer `unsqueeze`)
cannot work: the seq fold's state space is *operand broadcast minus
the step axis*, and a per-step `unsqueeze` against a larger state
(`(b,d)→(b,d,1)` vs a `(b,d,d)` accumulator) has no flat-space
meaning.  The sound commute is the other direction:

`view(select(child, dim=D, index=i))` ≡ `select(view'(child),
dim=D', index=i)` where `view'` re-bases the view's `dim` into
child coordinates and `D'` tracks the step axis through the
insert/remove.  `_seq_view_lift` walks an `unsqueeze`/`squeeze`
chain (innermost-first, `shape_of`-typed ranks) down to a `select`
and returns `(viewed_child, select_dim, index)`; `_seq_deindex`
then mints the operand metavar against the **viewed** child exactly
as before — `names`/`operands`/`events`, the identical-skeleton
check, the single-step-axis check and the index==position check are
all untouched.

Why this is the right frame: lifting grows the *operand's* shape so
the broadcast **includes the state dims**.  GLA's
`gg[:,t].unsqueeze(-1)` on a `(b,T,d)` operand becomes
`unsqueeze(gg, 3)` selected at dim 1 — a `(b,T,d,1)` sequence whose
broadcast with the `(b,T,1,d)` vv-view and `(b,T,d,1)` kk-view is
`(b,T,d,d)`, so `state = (b,d,d)` is exactly the recurrence's
matrix state.  Every downstream consumer is unchanged:
`shape_of` types `unsqueeze`/`squeeze` already;
`seq_scan_denotation`'s `broadcast_tensors`+`movedim` materialises
the views; `_seq_args` does the same for the Triton rows; the flat
kernel's `acc = a*acc + b` is elementwise over the materialised
`(b,d,d)` step slices.

## What mints now (zoo census, 16×8×2 fp64 intake)

| workload | before | after |
|---|---|---|
| RealGatedLRU | seq site, minted, pays ~0.29× | unchanged (spec identical) |
| GatedLinearAttention | spine found, `scan_spec` decline | **seq site minted + kept** |
| MambaSelectiveScan | spine found, `scan_spec` decline | **seq site minted + kept** |

Both new specs are `applyd(aff_diag(Q1, mul(Q2,Q3)), H,
{kind:"seq", axis:1})` — view-free skeletons; the views live in
`leaf_env` as `unsqueeze` Op terms the ordinary bindings evaluate.

## Measured — real intake → referee path, fp64 CUDA (RTX 2050)

Unmodified `_zoo_case` (`_mint_tiers` → `profitable`,
`kernel_for=_triton_kernel_for`, `denotation_for=
_claim_denotation_for`, reps=20):

- **GatedLinearAttention `scan_16`**: Triton scan tier bound
  (`gen` declined, `scan` bound); PROFITABLE_LOG spelled fold
  **226.1–244.7 µs vs Triton fold 78.6–96.3 µs → ratio ≈ 0.35–0.39×
  (~2.6–2.9× faster), KEPT**; `scan_16_k` in `sink.supported_ops`.
- **MambaSelectiveScan `scan_16`**: same path — spelled **186.2 µs
  vs Triton 69.7 µs → 0.375× (~2.7×), KEPT**.
- Soundness, both sites: `handler_denotation` (the fold) vs
  `eval_term` of the original unrolled chain on the real leaf
  values — **max abs diff 0.0** (exact, fp64).  Triton kernel vs
  fold denotation on CUDA: 2.2e-16 (fp64 rounding).

## Honest boundary (narrower than before, still real)

- Views lift **only directly over a `select`** (chains allowed).
  `unsqueeze(mul(select, select))` — a view over a compound step
  expr — keeps the view node in the skeleton and `scan_spec`
  declines at handler mint (tested).
- Grammar is `unsqueeze`/`squeeze` **with an int `dim`** only.
  `reshape`/`view`/`expand`/`permute` and dim-less `squeeze`
  decline; attr-carrying or untypeable (`shape_of` INVALID)
  children decline.
- The fold still requires the diagonal-affine `add(mul(a_t,h),b_t)`
  spine with per-step selects walking ONE axis in order — chunked/
  parallel recurrences and genuinely non-diagonal state updates
  remain out of scope (Mamba/GatedDeltaRule/DeltaRuleBlock/RWKV/
  chunked retention still mint nothing).
- Cost: lifted operands materialise the broadcast `(b,T,…,state…)`
  frame — for GLA/Mamba that's `(b,T,d,d)` vs the spelled
  `(b,T,d)` operands, i.e. the kernel pays O(T·b·d²) loads where
  the chain paid O(T·b·d) plus an outer product.  It still wins
  ~2.7× at 16×8×2 fp64 because the spelled fold is a T-step python
  loop; at larger T the launch amortises further, at tiny states
  the materialisation could flip it — the referee decides per site,
  as designed.

## Lever A rider — `deliver` is now a spec field

The Stage-2 leftover "no deliver hook in spec" closes as data:
`deliver` accepts `"auto"` (default — the board's own hook),
`"none"`, or `"torch"` (load-validated to `board="torch"`;
`true`/`false` spell `"auto"`/`"none"`).  `torch-silu.json` now
declares `"deliver": "torch"` explicitly.  Meta-arena stays
hookless under `"auto"` — symbolic probe terms have no tensors,
leaves or example input to lower+verify against; `"torch"` there
is refused at load with that reason.  Verified:
`play --domain-spec torch-silu --deliver` →
`{'delivered': True, 'verified': True, 'max_abs': 1.19e-07}`.

## Files

- `packages/mlopt-discovery/src/mlopt_discovery/play.py` —
  `_seq_view_lift` + the `_seq_deindex` view branch.
- `packages/mlopt-discovery/src/mlopt_discovery/domain_spec.py` —
  `deliver` field, `DELIVERS`, `_deliver_for` gating,
  `spec_summary` reports it.
- `packages/mlopt-discovery/src/mlopt_discovery/domains/torch-silu.json`
  — `"deliver": "torch"`.
- `tests/test_discovery_play.py` — GLA-shaped lift/fold/decline
  tests; `tests/test_domain_spec.py` — `test_deliver_field`.
