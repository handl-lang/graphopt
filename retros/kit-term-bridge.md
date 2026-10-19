# kit-term-bridge — mlopt terms emitted as `.hnl`, evaluated through CatoptKit

**Date:** 2026-10-20 · **Surface:** `packages/mlopt-kit/src/bin/hnl-run.rs`
(+ `tests/cli.rs`), `lab/kit/term_hnl.py`, `tests/test_kit_bridge.py` ·
**Gates:** `cargo test -p mlopt-kit` — **21/21 green**;
`pytest tests/test_kit_bridge.py` — **8/8 green** (marked `e2e`: the
suite crosses the external engine boundary).

## What landed

The frontier after `mlopt-kit` (retro `mlopt-kit.md`) was the second
half of "the e-graph's terms and the kit's runtime are the same program
space": mlopt's optimization machinery emitting/consuming programs
through the kit, not just hand-written `.hnl`. Three pieces, each thin:

- **`hnl-run`** — a subprocess door over `CatoptKit`: `.hnl` in on
  stdin or a file path, the `serde_json::Value` boundary form out
  (`{"__tensor__": …}` for a tensor). `--project <dir>` wires a
  `.handl.toml`; `--repeat N` evaluates the source N times inside one
  kit — the measurement hook a costed decision needs
  (`(t(N) − t(1))/(N−1)` wall per call). Deliberately *program-generic*
  — nothing term-aware — so emitted terms and the `hnl/` modules go
  through the same door.
- **`lab/kit/term_hnl.py`** — the emit/eval bridge. `emit` renders any
  term as a `tensor.*` expression from a **data table** (`_OpSpec` per
  op: `slots` assemble the call; `accepts` pins attrs the surface can
  honour — `conv2d` has no dilation slot, `batch_norm` has no
  `training=True`; `drop` ignores the eval-inert `momentum`/
  `cudnn_enabled`; `arity` picks the signature by operand count —
  `batch_norm`'s export variants synthesize the identity affine the op
  requires). `emit_program` binds `Var`/`Param` leaves in a `do`-
  prelude: `{"shape","data"}` → `reshape(from_list(flat))`, `{"hnl"}`
  → a verbatim expression (the marshaling channel for *computed*
  inputs — the fold's `s`). `KitRunner` shells `hnl-run` and decodes
  the wire form.
- **`tests/test_kit_bridge.py`** — the verification, three claims.

## Verified

1. **A real artifact end to end.** `TorchSource().to_ir` of an
   eval-mode `Conv2d → BatchNorm2d` — the actual `aten.conv2d` +
   `aten.batch_norm` IR with its full attr tail — emits as
   `tensor.batch_norm(tensor.conv2d(x, p_conv_weight, p_conv_bias,
   [1,1], [1,1]), …, 0.00001)` and evaluates through the kit to the
   torch-fp64 answer (**max abs diff 6.7e-16** on a seeded fixture).
2. **A law's two forms through the runtime.** `conv_channel_scale`
   (the `CONV_FOLD_RULES` anchor) applied via
   `meta.apply_rewrite_at`: `mul(conv2d(x,W,b), sv)` and its
   `conv2d(x, W∘sv′, b∘sv)` rewrite both emit and evaluate
   **dyadic-exact-equal** — and `dag_cost(flops_cost)` credits the
   fold (the absorbed `mul`/`reshape` are param-only subtrees —
   compile-time work).
3. **An optimization decision costed through the runtime.** The
   conv+BN fold as a decision: spelled
   `batch_norm(conv2d(x,W,b), g, be, m, v, eps)` vs deployed
   `conv2d(x, W2, b2)` — where `W2`/`b2` are *mlopt subterms*
   (`mul(w, reshape(s,(C,1,1,1)))`, `add(mul(b,s), sub(be,mul(s,m)))`)
   materialized through the kit itself. Deployed ==
   unfolded (< 1e-12), and the measured per-call cost is
   **82.1 ms vs 95.4 ms** (`--repeat 30`, debug build) with
   `dag_cost` agreeing directionally (**36864 vs 55296**). The
   fold's value lands where it should — in the deployed program.
4. **Generality + the honest bound.** A
   matmul/add/sub/reshape/transpose/concat term evaluates against the
   torch reference; `pack` emits a multi-output list; `sigmoid`,
   `rsqrt`, and training-mode `batch_norm` each refuse by name.

## The precisely-named missing surface

The one piece of the conv+BN fold the bridge cannot emit as a *term*:
`s = γ·rsqrt(σ²+ε)` — **elementwise unary ops are absent from
host-tensor's 18-op surface** (`rsqrt`/`sqrt`/`neg`/`exp`/`sigmoid`,
and the reductions `sum`/`mean`/`max`). The spelled RHS of the fold
composite dies on `rsqrt` first (`UnsupportedOpError("rsqrt")`, tested).
Workaround exercised here: `s` rides the `{"hnl"}` marshaling channel —
the `(C,)` stats materialize to lists and scalar `**` finishes each
channel, `hnl/conv_bn.hnl`'s own escape. The honest fix is a host-tensor
op addition (elementwise unaries), not bridge machinery.

## Findings worth carrying

- **The handl lexer does not read scientific notation.** `1e-05`
  lexes as the token sequence `1` `e` `-05` — *silently* evaluating to
  `1` in statement position (measured: `e :: 1e-05` bound `e = 1`),
  and failing inside a list. A mlopt `eps=1e-05` emitted naively
  would silently become `eps=1`. `_num` expands every float through
  `Decimal` (`1e-05` → `0.00001`), and the emitted-source test pins
  the spelling. This is an upstream lexer gap worth a handl-side fix.
- **`dag_cost`'s param-only discount models compile-time folding the
  kit runtime does not perform** — the emitted program evaluates
  `mul(w, reshape(sv))` per call. That is why the fold-decision test
  separates *compute* (the fold, run once) from *deploy* (folded
  weights as leaf data): the per-call measurement then prices what
  the cost model prices. Reported honestly: through the bridge, an
  unfolded `mul(conv,sv)` is *cheaper per call* than its folded RHS
  spelled as runtime work.
- **Bound names**: torch-export leaf names are dotted
  (`conv.weight`) and dots are field access in handl — `_sanitize`
  maps to `conv_weight`; `tensor` itself is reserved (binding it
  would shadow the op namespace).
- The upstream `hnl` CLI (`host-cli`) is **not** a usable door: it
  panics in `plugin-engine-typecheck-handl` on a trivial program and,
  more fundamentally, has no `host-tensor` `OpRegistry` linked — the
  `tensor.*` ops are Rust impls only `CatoptKit` wires. That is why
  `hnl-run` lives in `mlopt-kit`.

## What remains open (named)

- **The e-graph → `.hnl` loop is one direction deep.** The bridge
  emits terms *into* the kit; nothing yet lifts a `TaggedValue::Tensor`
  or an `.hnl` program *back* into IR (the `OnnxSource`/`OnnxSink`
  precedent says the seam shape is a `Source`/`Sink` pair — a
  `KitSink` implementing `ports.Sink.lower/verify` is the natural
  next step).
- **Promotion floor.** `lab/kit/term_hnl.py` sits in `lab/` under the
  99%-coverage promotion rule; its tests are `e2e`-marked, so the dev
  lane (`-m "not slow"`) runs them but `-m "not e2e"` does not.
- `--repeat` timing is wall-clock in a debug build — directionally
  honest, not a benchmark rig.
