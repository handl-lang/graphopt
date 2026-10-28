# Build list — what has to exist before the machine can find a kernel

**Status:** proposed · dependency-ordered · M1 scope only
**Definition of done:** the four `split_k` variants of `qgemv` are
*generated from IR terms* (not written by hand), all agree with
`awq_gemm` (cosine ≥ 0.9999), and at least one matches the hand-written
kernel's cold time on `down_proj` (58.9 µs).

| # | build | why it is needed first | home | effort | gate |
|---|---|---|---|---|---|
| **1** | **ADR: kernel IR enters as a `Language`, not as combinators** | the kernel's six combinators stay closed; the IR must be recorded as data or the next proposal quietly adds a combinator | `host/project/adrs/` | 1 h | ADR accepted |
| **2** | **`kit-cuda-lang`** — the IR as YAML: 8 ops, grades, cost hooks | the vocabulary the search ranges over; data, so a new op is a data change | new repo | ½ day | every op has a typed signature + grade row |
| **3** | **torch denotation per op** | equality becomes checkable; without it "the body is transparent" is a slogan | inside `kit-cuda-lang` tests | ½ day | each op has a reference function |
| **4** | **`compile-triton`** — IR → Triton, one op (`qgemv`) | the lowering that makes the term runnable; fixed variants first, general later | new repo (`compile-*` family) | 1–2 d | 4 variants compile and run |
| **5** | **Equality gate vs the incumbent** | the AWQ nibble-order bug survived because the reference shared the assumption; comparison must be against `awq_gemm`, never our own reimplementation | `graphopt/bench` + kit tests | ½ day | cosine ≥ 0.9999 for all variants |
| **6** | **Search driver in `kernelopt`** | enumerates IR terms, applies the equality gate, prices with the roofline bound, then measures with the deploy referee | `kernelopt` | 1 d | emits a best kernel + a routing row |

**Total to M1:** ~4 days, and nothing in the list is speculative — each
item has a home in a family that already exists.

## Deliberately not in M1

- **`lib-cuda` port + `lib-cuda-*` adapters.** Needed to *ship* the
  kernel through the platform, not to *find* it. Deferred until the search
  proves it can find one worth shipping.
- **M > 1 tiling / prefill, FP8, GPTQ layouts.** Coverage work; it only
  makes sense once one operator is proven.
- **Compile cache and pipelining combinators.** Both are required for
  *scale* (~10⁴ candidates) but not for the M1 gate (4 variants). Flagged
  so they are not forgotten: without a cache the search is rate-limited by
  Triton compilation.
- **Static cost ranking.** The model is used only as a hard bound
  (roofline floor, legality). This week's evidence: warm-cache ranking
  contradicted deploy ranking on three of four shapes.

## What the M1 result would settle

If a fixed lowering matches the hand kernel on one operator, generalising
is worth it and the search has a substrate. If it cannot, no amount of IR
generality will help — a codegen that loses to 30 lines of hand-written
Triton will lose at scale too, and the honest response is to keep
hand-writing kernels and use the search only for configs.
