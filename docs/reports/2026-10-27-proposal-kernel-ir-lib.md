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
| **kernel IR ops** (`tile_load`, `unpack_i4`, `dot`, `split`, `stage`, `reduce`, `store`) declared with grades and costs | data (`Language`) | new **`kit-cuda-lang`** | a language is data — `kit-prelude-lang` pattern |
| **lowering IR → Triton** | engine | new **`compile-triton`** | the `compile-*` family already exists (`compile-js`, `compile-llvm`, `compile-dot`) |
| **running a compiled kernel** | port + adapters | **`lib-cuda`** (contract) + **`lib-cuda-*`** (impls) | the `lib-X` / `lib-X-c` split the platform already uses |
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
| **M0** | IR declared in `kit-cuda-lang`; denotation + cost per op | every op has a torch reference and a byte count |
| **M1** | `compile-triton` lowers the four `split_k` variants | all four denotations equal to the incumbent's (cosine ≥ 0.9999) |
| **M2** | `kernelopt` searches the IR for `down_proj` | matches 2.96× or beats 42 % of roofline, structure not named by hand |
| **M3** | add `tile_load` with M>1 | covers prefill and `gate_proj` |

## Risks

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
