# experiments/ — the runnable proof

`mlopt-experiments` is the evidence-generation crate: each bin is one
experiment the paper cites. The handl stack is composed directly from
declared git deps (`src/lib.rs` assembles the kit — the same component
set mlopt-kit composes; the direct `mlopt-kit` cargo dep is unblocked
now that its deps float on the post-ADR-0128 `hnl-engine-*` crates at
current main — this crate keeps the composed form anyway, as evidence
that the stack assembles from declared components alone).
The `.hnl` artifact under measurement resolves in-tree — mlopt-kit is
a workspace member of this repo (`packages/mlopt-kit`); the
sibling-checkout / pkg-slot / fetched arms below keep a standalone
`research-mlopt` checkout working. Never vendored.

## `conv-bn-fold` — the founding result

Runs mlopt's conv+BN weight-space fold end to end through the handl
runtime and emits the two numbers the paper's §6 cites:

- **certification** — `conv_bn` (the spelled
  `batch_norm(conv2d(x, W, b), γ, β, μ, σ², ε)`) and `conv_bn_fold`
  (the minted law's folded RHS) evaluate to *value-equal* tensors —
  bitwise on the dyadic fixture — through the kit's real
  `lex → parse → eval` pipeline;
- **runtime pay** — per-call wall time of each arm over `--repeat N`
  evals in one kit (`hnl-run --repeat`'s measurement shape, in
  process).

`mlopt-experiments` is deliberately *not* a member of the root cargo
workspace (see the `exclude` row in `Cargo.toml` — it keeps its own
manifest + Cargo.lock so the experiment arm stays a standalone
composition of the same dep edges mlopt-kit declares), so cargo
invocations go through `--manifest-path`:

```bash
cargo run --release --manifest-path experiments/Cargo.toml --bin conv-bn-fold -- --repeat 200
cargo run --release --manifest-path experiments/Cargo.toml --bin conv-bn-fold -- --repeat 0   # certify only
cargo run --release --manifest-path experiments/Cargo.toml --bin conv-bn-fold -- --json       # machine-readable last line
```

## Resolving the artifact (no vendoring)

The program under measurement is mlopt-kit's `hnl/conv_bn.hnl`. The
bin resolves it by this ladder, first hit wins:

1. `--conv-bn <path>` / `CONV_BN_HNL` — explicit override;
2. `packages/mlopt-kit/hnl/conv_bn.hnl` — in-tree (this repo);
3. `$MLOPT_DIR/packages/mlopt-kit/hnl/conv_bn.hnl` — a named mlopt
   checkout (standalone arm);
4. the `.handl.toml` `[patch]` dir — `../mlopt/packages/mlopt-kit`
   sibling dev-link (inert where mlopt isn't an org sibling);
5. the `.handl.toml` git-dep arm's package-cache slot —
   `$HANDL_PKG_DIR/mlopt-kit` else `~/.handl/pkg/mlopt-kit` (the dir
   the ADR-0010 fetch hook will populate; a checkout/symlink is today's
   stand-in);
6. `experiments/.cache/mlopt-kit/hnl/conv_bn.hnl` — `make artifact`
   fetches it from the declared git source
   (`raw.githubusercontent.com/handl-lang/research-mlopt/main/…`) when no local
   arm resolves.

The imported program's own `from "host-tensor" import tensor` resolves
through *this repo's* `experiments/.handl.toml` — the `.hnl`-level dep
manifest (`host-tensor` + `mlopt-kit` declared by name, `[patch]` for
sibling checkouts).

## The Python-side evidence

`conv_fold_premise_pay_probe.py` (the gauntlet account — guard
transport, premise-anchored truth, typed pay, novelty) lives in mlopt
and is *invoked*, not copied:

```bash
cd "$MLOPT_DIR" && .venv/bin/python lab/probes/conv_fold_premise_pay_probe.py
```
