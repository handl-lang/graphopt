# TODO — graphopt

## title: TODO type: task date: 2026-09-26

## HIGH — ADR-0159: dissolve the optimizer domains into ext tools

## Adopted on three triggers, not a date ( host/project/adrs/0159-… ):

- [ ] T-002 the domain's public surface unchanged for four weeks

## HIGH — M1: the machine finds the kernel, not me (five items, ~5 days)

## Done — 2026-10-27 (kernel margin, valid)

## Active — HIGH PRIORITY (the open questions, post-plan-0032)

## Active

- [ ] T-009 Meta-arena Q3 ( project/plans/0021-meta-arena.md ) — measured, partially answered. Cold linear REINFORCE beats scripted on mean reward (−0.04 vs −0.10 at 2000 episodes) only via the degenerate extract-early policy; the composed declare→handle line was unreachable by sampling until the claim(tag) option move fused it (claim→extract wins silu 2→1 on held-out boards, +0.52 payouts) — reachability fixed, consolidation by the memoryless policy still open. Retro: project/retros/meta-arena-player.md . 2026-11-07: imitation+RL bake SHIPPED as mlopt\_discovery.meta\_prior.META\_ARENA\_PLAYER\_WEIGHTS (+0.17 vs macro prior +0.13 held-out; plays literal claim, op:macro −1.2). Curriculum measured negative; upstream gamekit \_observe boundary fix landed (stale tests updated). Next levers: richer policy (2nd-order feature / neural arm), claim-availability curriculum; spelled declare→saturate→handle line still unplayed.
- [ ] T-010 Generated kernels ( genkernel ) — the novelty question, measured. Handler entries minted for elementwise subterms the shipped HANDLERS never named; torch.compile 'd bindings; gen sink supports+lowers them; claim binds them — the board's vocabulary grows with lowerings it built. The concrete-soundness fix this forced ( \_canon\_concrete , claim\_tags ) is a real correction: skeleton-level coverage could bind mismatched kernel semantics. play --domain gen --probe : bigfuse (6-op chain) → claim(gen\_0) → verified module at 2.87× vs eager (6.1× at 256×1024); 2-op fusions honestly lose (~0.86×, compiled-call overhead). Retro: project/retros/gen-kernels.md . 2026-11-07: multi-claim composition pinned; size guard measured negative; OFFER-SIDE COST GUARD LANDED (\_claim\_cost\_gate — profitable referee at board build; chain\_mlp 1.68→3.44x vs compile, twosite 0.62→1.72x via dropping unpriced decayed\_0). Caveat: profitable noise is now offer-visible. Next: attr-bearing bodies (attr\_mvars machinery exists; site-collector attr preservation unmeasured), reduction bodies (seq-scan needs a carrier-aware spelled referee), two-sample referee hardening.

## Deferred

## Tech debt to fix

## Recovered — session 1792950048869\_3wjyq (generated-kernel claims)

## Recovered — session 1792996176495\_5kxw5 (AWQ kernel margin)
