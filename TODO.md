# TODO

## Active

- [x] ADR 0003 (evaluation as an independent dimension) — **accepted,
      in force**; all 12 stages of
      `project/plans/0016-evaluation-dimension.md` landed.

- [ ] Meta-arena Q3 (`project/plans/0021-meta-arena.md`) — **measured,
      partially answered.**  Cold linear REINFORCE beats scripted on
      mean reward (−0.04 vs −0.10 at 2000 episodes) only via the
      degenerate extract-early policy; the composed
      declare→handle line was unreachable by sampling until the
      `claim(tag)` option move fused it (claim→extract wins silu
      2→1 on held-out boards, +0.52 payouts) — reachability fixed,
      consolidation by the memoryless policy still open.  Retro:
      `project/retros/meta-arena-player.md`.  Next levers:
      curriculum boards, imitation warm-start, richer policy.

- [ ] Joint forward/backward optimization — `training.backward()`
      derives VJP programs; `joint_probe`/`--joint` puts fwd+grads
      in one e-graph: `linear_fwd` 13→10 (the VJP re-uses the whole
      forward term).  Silu/softmax flat — the fold laws hide the
      shared subterm; coordinated extraction measured a wash under
      count_cost (single-use sharing ≡ fusion).  Retro:
      `project/retros/joint-training-graph.md`.  Next: real
      training-corpus probes, combined fwd+bwd executor timing.

- [ ] The game layer is generic: `catopt_discovery.engine`
      (`Board` protocol + episode/train/eval drivers),
      `players.LinearPolicy` (featurizer-injected learner),
      `play` (domain registry + CLI).  Domains: `meta`, `joint`,
      `search`, `torch` (real `nn.Module`s under the sink bound;
      `--deliver` lowers the winning extraction to a verified
      runnable module).  Retro: `project/retros/generic-game-layer.md`.

- [ ] Generated kernels (`genkernel`) — **the novelty question,
      measured.**  Handler entries minted for elementwise subterms
      the shipped HANDLERS never named; `torch.compile`'d bindings;
      gen sink supports+lowers them; `claim` binds them — the
      board's vocabulary grows with lowerings it built.  The
      concrete-soundness fix this forced (`_canon_concrete`,
      `claim_tags`) is a real correction: skeleton-level coverage
      could bind mismatched kernel semantics.  `play --domain gen
      --probe`: `bigfuse` (6-op chain) → `claim(gen_0)` → verified
      module at **2.87× vs eager** (6.1× at 256×1024); 2-op fusions
      honestly lose (~0.86×, compiled-call overhead).  Retro:
      `project/retros/gen-kernels.md`.  Next: size-guarded claim
      offers, multi-claim composition, attr-bearing/reduction
      bodies.

## Deferred

- [ ] `paper/` — a short paper-style write-up of the research claim:
      categorical semantics exposes transforms conventional tensor-level
      optimizers don't reach efficiently. Pull numbers from
      `bench/reassoc_scale.py` (head-to-head vs Inductor at scale) and
      `bench/search_efficiency.py` (alternatives-space vs e-graph size).
      Audience: compilers/PL reviewers. Frame around the fair-comparison
      pipeline (same backend, real checkpoints) — cite REPORT.md on this
      branch for the existing numbers and the falsified directions.
      Unblocked (both benches landed); user wants it deferred.

- [ ] `plan 0002` leftovers — Phase E recognizer consolidation
      (law-side e-class bindings vs lowerer-side extracted terms —
      substrates genuinely differ, revisit only if a shared kernel
      emerges); plan-dict typing stays dict-of-any.

## Done

- [x] `bench/reassoc_scale.py` — landed. Post-grad FX capture proves
      Inductor keeps k left-assoc mms; CatOpt folds weights into one
      param → 1 mm. 8.93× vs Inductor at k=8, 16.12× at k=16 (CPU).
- [x] `bench/real_linear_attn.py` — landed. RetNet/GLA/delta-rule
      blocks: scan lift + value-chain fold verified fp64-exact;
      retnet 2.8–4.8× vs eager at the closed-form floor. Honest CPU
      negatives vs Inductor (pointwise fusion wins; launch-bound
      devices invert — run --device cuda for the real story).
- [x] `bench/search_efficiency.py` — landed. Saturated e-graph encodes
      all Catalan(k−1) bracketings (verified: 16,796 in 231 e-nodes at
      k=11); bounded saturation ≥10^5.6 programs in ~0.5s at k=24.

- Engineering: monorepo split (5 domain packages), 100% coverage
  line+branch, ruff/pyright clean, 1,585 tests, review pass fixed 5
  audit findings (incl. the omd_lower dense-fiber broadcast bug).
