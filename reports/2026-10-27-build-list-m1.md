# Build list — what has to exist before the machine can find a kernel

**Status:** proposed · dependency-ordered · M1 scope only
**Refactor context:** lab-form per ADR-0159; ADR-0160 allocates the IR.
**Visibility (ADR-0150):** `lib-kernel-ir` public; `compile-triton`, `lib-triton`,
`ext-opt-*` private until done.
**Definition of done:** the four `split_k` variants of `qgemv` are
*generated from IR terms* (not written by hand), all agree with
`awq_gemm` (cosine ≥ 0.9999), and at least one matches the hand-written
kernel's cold time on `down_proj` (58.9 µs).

| # | build | why it is needed first | home | effort | gate |
|---|---|---|---|---|---|
| **1** | **ADR-0160: the kernel IR is `Term` + kit-declared ops** | the kernel's six combinators stay closed; the IR enters as data, and the decision must be recorded where the IR lives | `opt-kernel` (today `kernelopt`) | 1 h | ADR accepted |
| **2** | **`lib-kernel-ir`** — the vocabulary as a handl *lib*: nine ops, grades, costs, **and a torch denotation per op** | the vocabulary the search ranges over; denotations make "equal meaning" checkable instead of a slogan | new repo — **public** | 1 d | every op has a signature, a grade row, a cost, a reference function |
| **3** | **`compile-triton`** — Rust engine: `Term` → Triton, for **`qgemv` and `gate_proj`**, 4 `split_k` variants each | the lowering that makes terms runnable; two opposite geometries stop it overfitting one shape | new repo — **private** (tooling tier, ADR-0150) | 2 d | 8 variants compile and run |
| **4** | **Equality gate vs the incumbent** | the AWQ nibble-order bug survived because the reference shared the assumption; comparison must be against `awq_gemm`, never our own reimplementation | `graphopt/bench` | ½ day | cosine ≥ 0.9999 on every variant |
| **5** | **Search driver in `kernelopt`** | enumerates `Term`s, applies the equality gate, prices with the roofline bound, then measures with the deploy referee | `kernelopt` | 1 d | emits a best kernel per operator + a routing row |

**Total to M1:** ~5 days.

**No kit in M1.** A kit assembles several libs; one lib is loaded
directly. `kit-kernel-ir` arrives at M2, when `lib-triton-qgemv` (the
binding), the cost tables and the base types have to meet — the same
reason `kit-prelude-lang` exists and a lone lib does not.

## Deliberately not in M1

- **`lib-triton-qgemv` binding.** Needed to *ship* the
  kernel through the platform, not to *find* it. Deferred until the search
  proves it can find one worth shipping.
- **M > 1 tiling / prefill, FP8, GPTQ layouts.** Coverage work; it only
  makes sense once both operators are proven.
- **Compile cache and pipelining combinators.** Both are required for
  *scale* (~10⁴ candidates) but not for the M1 gate (4 variants). Flagged
  so they are not forgotten: without a cache the search is rate-limited by
  Triton compilation.
- **Static cost ranking.** The model is used only as a hard bound
  (roofline floor, legality). This week's evidence: warm-cache ranking
  contradicted deploy ranking on three of four shapes.

## What the M1 result would settle

If a fixed lowering matches the hand kernel on `qgemv` and reaches parity
on `gate_proj`, generalising is worth it and the search has a substrate.
If it cannot, no amount of IR generality will help — a codegen that loses to 30 lines of hand-written
Triton will lose at scale too, and the honest response is to keep
hand-writing kernels and use the search only for configs.
