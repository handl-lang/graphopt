# TODO — graphopt

## title: TODO type: task date: 2026-09-26

## HIGH — ADR-0159: dissolve the optimizer domains into ext tools

## Adopted on three triggers, not a date ( host/project/adrs/0159-… ):

- [ ] T-001 a caller that is not the researcher needs the capability
- [ ] T-002 the domain's public surface unchanged for four weeks
- [ ] T-003 the referee is shared ( bench/benchkit usable by any domain)

## HIGH — M1: the machine finds the kernel, not me (five items, ~5 days)

## Done — 2026-10-27 (kernel margin, valid)

## Active — HIGH PRIORITY (the open questions, post-plan-0032)

## Active

- [ ] T-009 Meta-arena Q3 ( project/plans/0021-meta-arena.md ) — measured, partially answered. Cold linear REINFORCE beats scripted on mean reward (−0.04 vs −0.10 at 2000 episodes) only via the degenerate extract-early policy; the composed declare→handle line was unreachable by sampling until the claim(tag) option move fused it (claim→extract wins silu 2→1 on held-out boards, +0.52 payouts) — reachability fixed, consolidation by the memoryless policy still open. Retro: project/retros/meta-arena-player.md . Next levers: curriculum boards, imitation warm-start, richer policy.
- [ ] T-010 Generated kernels ( genkernel ) — the novelty question, measured. Handler entries minted for elementwise subterms the shipped HANDLERS never named; torch.compile 'd bindings; gen sink supports+lowers them; claim binds them — the board's vocabulary grows with lowerings it built. The concrete-soundness fix this forced ( \_canon\_concrete , claim\_tags ) is a real correction: skeleton-level coverage could bind mismatched kernel semantics. play --domain gen --probe : bigfuse (6-op chain) → claim(gen\_0) → verified module at 2.87× vs eager (6.1× at 256×1024); 2-op fusions honestly lose (~0.86×, compiled-call overhead). Retro: project/retros/gen-kernels.md . Next: size-guarded claim offers, multi-claim composition, attr-bearing/reduction bodies.

## Deferred

## Tech debt to fix

- [ ] T-016 synop real delivery \* — docs/reports/prim\_delivery.md measured: folds compress terms (6993→4353 ops on GLA-large) but the spec-walk binding ( genkernel.\_eval\_spec ) loses to spelled eager wall-clock on every clean case; the cost-model 0.5–0.8× pay is not delivered. Real delivery needs a fused-body codegen path for minted ops, not a spec interpreter. First step, measured 2026-10-27: the mint does register its ops ( mlopt\_kernel.sink.\_table() 84→86, ksynop\_9e3cb9a03a , ksynop\_efc263c3dd ), but the zoo sweep's delivery path is play.gen\_probe → \_gen\_board(case) and that board's bank comes from genkernel — it never reads the sink table. Teach \_gen\_board to merge the minted ksynop\_\* folds into its bank, then re-sweep the 46 losing zoo rows (median 0.979×). · status:in-progress · note:2026-11-07 update — fused Triton path landed (p26) + strided-load fix (p29 §3) — broadcast materialization removed; honest boundary measured: elementwise prim folds pay iff output ≳0.5M elems fp64, all zoo sites are 256–32k → gate refuses them; attr-metavar prims still lose via compiled spec-walk (needs unrolled codegen). Also fixed referee contamination: profitable now drains inductor compile workers before timing. OPEN: fp32 loser re-sweep; attr-metavar codegen path.

## Recovered — session 1792950048869\_3wjyq (generated-kernel claims)

## Recovered — session 1792996176495\_5kxw5 (AWQ kernel margin)
