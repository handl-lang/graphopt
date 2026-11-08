# TODO — graphopt

## title: TODO type: task date: 2026-09-26

## HIGH — ADR-0159: dissolve the optimizer domains into ext tools

## Adopted on three triggers, not a date ( host/project/adrs/0159-… ):

- [ ] T-002 the domain's public surface unchanged for four weeks

## HIGH — M1: the machine finds the kernel, not me (five items, ~5 days)

## Done — 2026-10-27 (kernel margin, valid)

## Active — HIGH PRIORITY (the open questions, post-plan-0032)

## Active

- [ ] T-009 Meta-arena Q3 ( project/plans/0021-meta-arena.md ) — measured, partially answered. Cold linear REINFORCE beats scripted on mean reward (−0.04 vs −0.10 at 2000 episodes) only via the degenerate extract-early policy; the composed declare→handle line was unreachable by sampling until the claim(tag) option move fused it (claim→extract wins silu 2→1 on held-out boards, +0.52 payouts) — reachability fixed, consolidation by the memoryless policy still open. Retro: project/retros/meta-arena-player.md . 2026-11-07: imitation+RL bake SHIPPED as mlopt\_discovery.meta\_prior.META\_ARENA\_PLAYER\_WEIGHTS (+0.17 vs macro prior +0.13 held-out; plays literal claim, op:macro −1.2). Curriculum measured negative; upstream gamekit \_observe boundary fix landed (stale tests updated). 2026-11-07 remainder measured: claim-availability curriculum (--availability) and the second-order feature arm (META_SPEC_V2, x:seq/x:seqh last-move x catalog, --prev-features) both LANDED and both FLAT — held-out identical (+0.17 learned-greedy, 20/40 declared-use) because the policy already sits at the oracle ceiling (gap ~-0.0001). Spelled declare->saturate->handle->extract line is reachable and pays (~+0.30 vs claim->extract ~+0.40 on silu) but oracle argmax picks it 0/400 — dominated wherever claim exists; --oracle-spelled/--spelled-demo-epochs landed as demonstration machinery. Neural arm not re-run (open-domain record already below linear). Remaining open is CANDIDATE side, not policy: oracle lines beyond the current claim/saturate/extract set.
- [ ] T-010 Generated kernels ( genkernel ) — the novelty question, measured. Handler entries minted for elementwise subterms the shipped HANDLERS never named; torch.compile 'd bindings; gen sink supports+lowers them; claim binds them — the board's vocabulary grows with lowerings it built. The concrete-soundness fix this forced ( \_canon\_concrete , claim\_tags ) is a real correction: skeleton-level coverage could bind mismatched kernel semantics. play --domain gen --probe : bigfuse (6-op chain) → claim(gen\_0) → verified module at 2.87× vs eager (6.1× at 256×1024); 2-op fusions honestly lose (~0.86×, compiled-call overhead). Retro: project/retros/gen-kernels.md . 2026-11-07: multi-claim composition pinned; size guard measured negative; OFFER-SIDE COST GUARD LANDED (\_claim\_cost\_gate — profitable referee at board build; chain\_mlp 1.68→3.44x vs compile, twosite 0.62→1.72x via dropping unpriced decayed\_0). Caveat: profitable noise is now offer-visible. 2026-11-07 remainder: attr bodies — _canon_taken equality fix landed (unhashable attr canon crashed gen_/decayed_ dedup); attr specs mint + torch binding verified max-diff 0.0; Triton declines attr nodes (compiled fallback only); zoo census = ZERO attr-carrying elementwise ops in 68 workloads — occurrence-blocked. Reduction bodies — no tier collects reduction specs; 4 decayed minted zoo-wide, prior 'kept' row not reproducible (referee noise); next step is a reduction TIER, not pricing. Seq-scan pricing — BLOCKED: no spelled fold denotation exists (ambient applyd = one map app; _IR_TO_TORCH has no fold op); moot today since _scan_sites pins kind=unrolled (seq=0/87); ALSO fixed a latent soundness gap — _zoo_case/_claim_cost_gate could have offered/delivered seq scans under map semantics via gen_sink; _claim_measurable now shared and seq scans dropped at both gates. Remaining: two-sample referee hardening; a reduction tier if a paying site appears.

## Deferred

## Tech debt to fix

## Recovered — session 1792950048869\_3wjyq (generated-kernel claims)

## Recovered — session 1792996176495\_5kxw5 (AWQ kernel margin)
