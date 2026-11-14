# What you can optimize — and what "faster" honestly means

<!-- provenance: graphopt:docs/reports/{trajectory_scan_impl,seqt_readout,zoo_corpus_expansion,cumsum_carrier,seqs_pay_sweep,module_loss_gla_mamba}.md + docs/site/how-to/optimize-a-model.md -->

The pipeline is: **lift a tensor program into IR → search equivalent
forms → prove the rewrite → measure against the real incumbent → emit a
runnable kernel.** This page states what that buys today and where it
stops — the scope is measured, not aspirational.

## The claim, precisely

For a model whose computation reduces to the covered tensor vocabulary,
you can lift it, optimize it, and receive a **verified** faster
equivalent — not a maybe-correct generated kernel, but one whose output
is checked against an independent denotation before it ships.

Measured examples (all delivered + verified on CUDA, `zoo_chain`
probes):

| workload class | result |
|---|---|
| Scan/recurrence modules (Mamba, GLA, delta-rule, headed/bidirectional scans) | **2–6×** vs eager at module level (`seqt` fused read-out tier) |
| Dispatch-bound plan tapes | graph capture beats `torch.compile` **7/8** re-swept frontier rows |
| Reduction/softmax carriers | paying corner exists (fp64 broadly; one fp32 fused-softmax case 1.255×) |

## What "covered" means

- **Vocabulary**: the tensor surface is ~31 ops (`host-tensor` /
  `lib-tensor` parity). Models built from matmul, attention, norms,
  scans, convolutions, elementwise chains lift today; exotic custom
  fused ops need a port first.
- **Corpus**: the measured zoo is **73 workloads**. Recognition
  machinery (collectors, `domain_spec`) generalizes, but pay is only
  *claimed* where the corpus measured it — new shapes go through the
  same verify-then-pay gauntlet.
- **Shape**: wins concentrate where structure exists — scans,
  reductions, dispatch-bound graphs. Plain elementwise workloads
  mostly tie `torch.compile`; that is a measured boundary, not a bug.

## What "faster" is checked against

The referee times candidates against the real incumbent — eager, the
spelled module, `torch.compile` — with same-executor arms so routing
alone cannot fake a win. Every kept kernel re-verifies numerically;
every decline is recorded. Honest negatives are part of the output:
`cumsum`/`cumprod` scans are covered and bit-exact but lose to cub
everywhere probed; trajectory-returning scans stay on the `seqs`
tier rather than claiming `seqt`.

## The path for your model

1. Express it in the tensor IR (Handl/`mlopt_torch` export).
2. `opt.optimize(model, x)` — search, lower, verify ([how-to](../how-to/optimize-a-model)).
3. For a whole new domain shape, a `domain_spec` JSON generates the
   playable domain and its own coverage ([define a domain](../how-to/define-a-domain)).

Prior infrastructure — the `kernel` term calculus and `host` runtime
semantics — is upstream of all of this; graphopt consumes it as a
pinned substrate.
