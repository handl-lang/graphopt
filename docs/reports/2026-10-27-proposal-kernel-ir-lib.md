# Proposal — a kernel IR as data, and the libs it needs

**Status:** proposed · follows `2026-10-27-proposal-int4-kernel-vocabulary.md`
**Question it answers:** if a kernel body should be transparent (a term,
not an opaque `triton.jit` blob), where does that term live and what
carries it?

## The rule that constrains the answer

The kernel's six combinators stay closed. A kernel IR is **not** a new set
of kernel combinators — that is exactly the "cost of an open kernel" the
kernel paper argues against. It enters the platform the way every other
language does: **as a `Language` (data), interpreted by engines.** So the
IR is a language, the search is over its terms, and nothing downstream of
the kernel grows.

## Where each piece lives

| artifact | kind | home | why there |
|---|---|---|---|
| **kernel IR ops** (`tile_load`, `unpack_i4`, `dot`, `split`, `stage`, `reduce`, `store`) declared with grades and costs | data (`Language`) | new **`lib-kernel-ir`** | a language is data — `kit-prelude-lang` pattern |
| **lowering IR → Triton** | engine | new **`compile-triton`** | the `compile-*` family already exists (`compile-js`, `compile-llvm`, `compile-dot`) |
| **running a compiled kernel** | port + adapters | **`lib-triton`** (contract) + **`lib-triton-*`** (impls) | the `lib-X` / `lib-X-c` split the platform already uses |
| **schedule terms, cost model, certificate, search** | domain | **`kernelopt`** | it already owns `Kernel_RULES`, the sink, the cost model and the sieve |
| **deploy-regime referee, routing** | measurement | **`graphopt`** | `bench/benchkit/regime.py` lives there |

Nothing new is invented beyond the IR itself; every other artifact lands in
a family that already exists.

## The IR, kept deliberately small

Ten ops, each with a **denotation** (a torch reference, so equivalence is
checkable) and a **cost** (bytes, ops, launches — the enrichment):

```
tile_load   (tensor, idx)        -> tile
unpack_i4   (packed, shift)      -> int4 tile
dequant     (q, zero, scale)     -> fp tile
dot         (a, b)               -> tile
split       (k_range, n_parts)   -> k_range
stage       (partial)            -> partial
reduce      (partials)           -> tile
store       (tile, idx)          -> ()
bias        (tile, bias)         -> tile
```

The test of "is the body transparent" is then mechanical: two terms are
equal iff their denotations are equal. `split_k = 4` becomes four `split`
terms with provably equal composed denotation, not a parameter someone
hopes is safe.

## Milestones

| # | deliverable | gate |
|---|---|---|
| **M0** | IR declared in `lib-kernel-ir`; denotation + cost per op | every op has a torch reference and a byte count |
| **M1** | `compile-triton` lowers the four `split_k` variants | all four denotations equal to the incumbent's (cosine ≥ 0.9999) |
| **M2** | `kernelopt` searches the IR for `down_proj` | matches 2.96× or beats 42 % of roofline, structure not named by hand |
| **M3** | add `tile_load` with M>1 | covers prefill and `gate_proj` |

## Review — what is wrong with the above

**1. The IR as specified cannot express what makes kernels fast.** The op
list has `tile_load`, `dot`, `split`, `stage`, `reduce` — all *data*
operations. It has no software pipelining, no accumulator reuse across K
tiles, no shared-memory staging. Those are the things that separate 42 %
of roofline from 70 %. A schedule language that cannot state them can only
express naive schedules, which will lose to the hand kernel every time.
**Minimum additions:** a `pipeline(depth)` combinator and `stage_shared`
(a residency annotation), or the IR is decorative.

**2. Denotation equality proves meaning, not schedule equivalence.** Two
terms with identical denotations can differ 3× in cost — that is the whole
point of searching. So "certify the body" reduces to the law we already
have; the IR's real payoff is *search*, and the review should say so
instead of implying it buys proof.

**3. A static cost model will prune winners.** Evidence from this week:
the grid search's own warm-cache ranking contradicted the deploy-regime
ranking on three of four shapes. A cost model is a cruder ranking than a
measurement, so it must be used only as a **hard bound** (roofline lower
bound, legality), never to choose between feasible candidates.

**4. Search is rate-limited by Triton compilation, not by ideas.** 288
candidates ≈ minutes; 10⁴ candidates ≈ hours of compilation. The proposal
does not mention a compile cache or a way to instantiate without a full
`triton.jit` round trip. Without that, the IR cannot be searched at the
scale where it beats hand-writing.

**5. The economics are unstated.** Today's result — 29 % end-to-end —
came from a hand-written kernel plus a config sweep. The IR is justified
*only* if it then covers many shapes and formats. If the plan is "a few
more kernels", the IR is over-engineering; if the plan is "hundreds of
shapes across GPTQ/AWQ/FP8", it is the only thing that scales. That
decision should be explicit before M0.

**6. Legitimacy, not just design.** A new `Language` in the platform
needs an ADR in the host repo — the kernel stays closed and the IR enters
as data, which is the right shape, but it should be recorded as such or
the next proposal will quietly add a combinator.

### Recommended cut-down for M1

Do **not** build the ten-op IR first. Build **one op end to end**: `qgemv`
with a fixed lowering, its torch denotation, and the four `split_k`
variants. Gate: all four denotations equal to `awq_gemm` (cosine ≥
0.9999) *and* at least one variant matching the hand kernel's speed.

If that works, the IR earns its second op. If the fixed lowering cannot
match a hand-written kernel on one op, generalising it is wasted work.


- **IR creep.** Ten ops is the budget; an IR that grows into a compiler
  project loses to just writing Triton. Anything that cannot be expressed
  in these ten is a finding about the platform.
- **Lowering quality.** `compile-triton` must reach the same code quality
  as the hand kernel, or the IR has bought provability at the cost of
  speed — measured, not assumed.
- **Residual opacity, stated once:** the *silicon's response* (registers,
  bank conflicts, ptxas) is never derivable from a term. Cost stays
  model + measurement. That is why the referee is part of the design and
  not an afterthought.
