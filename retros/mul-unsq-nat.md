# Retro: admitting `census:mul_unsqueeze` — the guard question, resolved

Date: 2026-10-14
Context: `exposed-math-discovery.json` recorded the exposure agent's
best find — `mul(unsqueeze U d, unsqueeze V d) = unsqueeze(mul U V, d)`
on `DeltaRuleBlock`: new, non-derivable, 4 typed fires, −12.27%
(992496→870723), verify pass, certificate replays — blocked only by
"conditional truth (view-oracle): no single mechanical feature
separates".  This retro records the admission decision: what the guard
is, why the machine-minted clause was not it, and what shipped.

## The truth region

`unsq(x, d)` inserts a size-1 axis at `d mod (rank+1)`; broadcasting
right-aligns.  Pushing `mul` through a shared insertion is the
naturality `mul(unsq u d, unsq v d) = unsq(mul u v, d)`.  The semantic
boundary is exact and mechanical:

- **equal ranks** — always true (both operands insert at the same
  normalized position; well-typedness of either side is the same
  condition: `bcast(u,v)` must resolve);
- **unequal ranks** — true iff each operand's flat-read coefficient
  map agrees across the two orders.  The shorter operand's inserted
  axis lands at `d + (R − r)` in the result grid vs the RHS's `d mod
  (R+1)`; inserts inside the broadcast pad (`d ≤ 0` region) commute,
  mid-rank inserts do not — e.g. `u=(2,), v=(1,2,2), d=1` equalizes
  the *grids* `(1,1,2,2)` both ways but reads `u[i]` vs `u[j]`.

Adversarial sweep (pure IR semantics — mod-normalized dims, numpy
broadcast; ranks ≤3, extents {1,2,3}, d ∈ −8..8): **12956 evaluable
equal sites, 2464 unequal, 1388 rhs-err**.  Verdicts:

| guard | bad-accepts | equal-missed |
|---|---|---|
| `rank-eq(u,v) ∧ shaped(bcast u v)` | 0 | 6156 (~48%) |
| grid-eq only — `bcast-eq(unsq-out u d, unsq-out v d, unsq-out(bcast u v) d, ·)` | **212** | 0 |
| `flat-map-unsq(u, UD, v, unsq-out(v, UD))` | **0** | **0** |

`flat-map-unsq` — the `wl`-naturality predicate minted for
`mul(unsq u K, reshape v S) = unsq(mul u v, K)` — is the *exact* region
when its partner-shape argument `S` is instantiated with the partner's
own `unsq-out` spec: it checks grid coincidence AND both operands'
flat-read maps.  That is the shipped guard — one clause, pure DSL
data, serializable.

## What auto_cond minted — and why it is not the guard

`auto_cond_object` on the bare pattern (synth_limit 360, 174 measured
envs: 69 equal / 16 bad) minted a single clause — the grid-eq
`bcast-eq` above — separating its measured domain perfectly and
**unsound** past it: the synthesized env bank never produced the
unequal-rank mid-insert corner (the sweep shows 212 bad-accepts).
The stock `_pred_bank` emits `flat-map-unsq` only with `reshape-out`
partner specs — the `unsq-out` instantiation the two-sided law needs
is outside the minted vocabulary, so the search picked the best
*available* clause and the corner is unmeasured, not unguardable.
Honest lesson for the bank: `flat-map-unsq`'s `S` position should
also range over `unsq-out` specs when the partner view is unsqueeze.

## The sibling: `mixed:mul_unsqueeze_r_id` — already admitted

`mul(V, unsqueeze(U,d)) → mul(U,V)` is α-identical to the shipped
`mul_unsq_pad_r` (promoted-laws.md row 6) — the same `unsq:d_in_pad`
family.  Corpus-wide tally of its 22 strict-match sites / 12 models:

- **2 sites accepted by the shipped guard** (intake:BroadcastPadRight,
  zoo:TimeCondConv) — true pad strips;
- **10 rejected sites are numerically unequal** — including all 4
  DiagDenseSSM sites, where `V` binds `b_eye`: `I·unsq(U,−1)` and
  `U·I` coincide *only* because the identity masks off-diagonals
  (`diag(U)` both ways) — a contextual equality, not a law.  The
  −14.3% "pay" measured there was the false direction made invisible
  by eye-masking; `unsq:d_in_pad` rejecting `d=−1` (trailing insert,
  not a pad) was the correct refusal;
- **10 rejected sites unevaluable** (err);
- **0 rejected-but-equal** — the shipped guard loses nothing.

So the same guard resolution *already* unlocks it: the store object
stays unshipped *as a store object* because the law it names is the
law that shipped.

## What shipped

`mul_unsq_nat` → `SIMPLIFICATION_RULES` (default-on; the pad-strip
family's home):

    mul(unsq u UD, unsq v UD) -> unsq(mul u v, UD)
    cond = ("flat-map-unsq", "u", "UD", "v", ("unsq-out", "v", "UD"))

Closure measurement (7-model slice, lone-rule reach): zero enode
growth off-site; `DeltaRuleBlock` +1.3% enodes (558→565), −12.3%
cost, `add_cert=pass`, fixed-point.  The default set is justified —
no speculative e-node bloat, real pay on-site.

`ALL_RULES` 82 → 83; `MACHINE_SHIPPED` 10 → 11; serialization census
56/56 cond-carrying rules round-trip; `missing_hooks == ()`.
