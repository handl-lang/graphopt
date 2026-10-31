# TODO — graphopt

## title: TODO type: task date: 2026-09-26

## HIGH — ADR-0159: dissolve the optimizer domains into ext tools

## Adopted on three triggers, not a date ( host/project/adrs/0159-… ):

- [ ] T-001 a caller that is not the researcher needs the capability
- [ ] T-002 the domain's public surface unchanged for four weeks
- [ ] T-003 the referee is shared ( bench/benchkit usable by any domain)

## HIGH — M1: the machine finds the kernel, not me (five items, ~5 days)

- [ ] T-004 in opt-kernel (today kernelopt ): the kernel IR is Term + kit-declared ops
- [ ] T-005 lib-kernel-ir (public): nine ops, grades, costs, one torch denotation per op — the vocabulary, and the equality reference
- [ ] T-006 compile-triton (Rust, compile-\* family): Term → Triton, qgemv + gate\_proj × split\_k ∈ {1,2,4,8}
- [ ] T-007 Equality gate vs awq\_gemm in graphopt/bench — cosine ≥ 0.9999 on every variant, inside the timing path
- [ ] T-008 Search driver in kernelopt : enumerate Term s → gate → roofline bound → deploy-referee measure → best kernel + routing row

## Done — 2026-10-27 (kernel margin, valid)

## Active — HIGH PRIORITY (the open questions, post-plan-0032)

## Active

- [ ] T-009 Meta-arena Q3 ( project/plans/0021-meta-arena.md ) — measured, partially answered. Cold linear REINFORCE beats scripted on mean reward (−0.04 vs −0.10 at 2000 episodes) only via the degenerate extract-early policy; the composed declare→handle line was unreachable by sampling until the claim(tag) option move fused it (claim→extract wins silu 2→1 on held-out boards, +0.52 payouts) — reachability fixed, consolidation by the memoryless policy still open. Retro: project/retros/meta-arena-player.md . Next levers: curriculum boards, imitation warm-start, richer policy.
- [ ] T-010 Generated kernels ( genkernel ) — the novelty question, measured. Handler entries minted for elementwise subterms the shipped HANDLERS never named; torch.compile 'd bindings; gen sink supports+lowers them; claim binds them — the board's vocabulary grows with lowerings it built. The concrete-soundness fix this forced ( \_canon\_concrete , claim\_tags ) is a real correction: skeleton-level coverage could bind mismatched kernel semantics. play --domain gen --probe : bigfuse (6-op chain) → claim(gen\_0) → verified module at 2.87× vs eager (6.1× at 256×1024); 2-op fusions honestly lose (~0.86×, compiled-call overhead). Retro: project/retros/gen-kernels.md . Next: size-guarded claim offers, multi-claim composition, attr-bearing/reduction bodies.

## Deferred

- [ ] T-011 plan 0002 leftovers — Phase E recognizer consolidation (law-side e-class bindings vs lowerer-side extracted terms — substrates genuinely differ, revisit only if a shared kernel emerges); plan-dict typing stays dict-of-any.

## Tech debt to fix

- [ ] T-012 Vendored probe code — copied third-party code in bench/drivers/probes/ (and bench/limits/registry.py duplicated into kernelopt during the split). Replace with a declared dep/submodule or delete if dead.
- [ ] T-013 Import-time domain registration — domains seat themselves via import side-effects ( mlopt\_kernel.play mutating DOMAINS /registry on import). Replace with explicit registration at call site.
- [ ] T-014 Pinned test counts — tests asserting literal rule/law/bank counts ( test\_prior\_art , test\_laws\_structure totals, AGENTS.md counts). Replace with derived counts or invariant-based assertions so bank growth doesn't churn tests.
- [ ] T-015 radon baseline — bilinear\_game.py flagged on clean main; regenerate scripts/complexity\_baseline.json or justify.
- [ ] T-016 synop real delivery \* — docs/reports/prim\_delivery.md measured: folds compress terms (6993→4353 ops on GLA-large) but the spec-walk binding ( genkernel.\_eval\_spec ) loses to spelled eager wall-clock on every clean case; the cost-model 0.5–0.8× pay is not delivered. Real delivery needs a fused-body codegen path for minted ops, not a spec interpreter. First step, measured 2026-10-27: the mint does register its ops ( mlopt\_kernel.sink.\_table() 84→86, ksynop\_9e3cb9a03a , ksynop\_efc263c3dd ), but the zoo sweep's delivery path is play.gen\_probe → \_gen\_board(case) and that board's bank comes from genkernel — it never reads the sink table. Teach \_gen\_board to merge the minted ksynop\_\* folds into its bank, then re-sweep the 46 losing zoo rows (median 0.979×).
- [ ] T-017 fragment-tower migration — local plan in project/plans/0033-fragment-tower.md + host ADR-0157; phases 0–5 (fragment dir → differential backend → denotations as models → fragment-emitting mint → fragment-move play → handl-native saturation).

## Recovered — session 1792950048869\_3wjyq (generated-kernel claims)

- [ ] T-018 Instrument why \_triton\_kernel\_for never binds — 8/8 on the 4 claim-bearing workloads; log kernels attempted/rejected/why, both dtypes. If the Triton path silently never fires, the "generated kernel beats spelled" claim has no measurement.
- [ ] T-019 Fold cost gate in profitable — two measured misses justify it: RMSNormPerHead fp64 0.21×, SigmoidAttention fp32 0.72×.
- [ ] T-020 9-round re-measure of the 5 straddlers — CapsuleRouting fp64; WaveNetGate , HyenaLongConv , SkipInitNet , DilatedBlockAttention fp32.
- [ ] T-021 Fold the fp32 sweep into the default driver path / CI table — so the quoted number can't drift back to the fp64 convention.

## Recovered — session 1792996176495\_5kxw5 (AWQ kernel margin)

- [ ] T-022 Push down\_proj from 42% toward 70% of the memory roofline — ~2.4× still on the table. (Redo-e2e landed as docs/reports/p18\_e2e\_29pct.md ; gate\_proj parity is already the M1 target above; "substrate-native" is M1 itself.)
