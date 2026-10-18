# experiments/ — the runnable proof

`mlopt-experiments` is the evidence-generation crate: each bin is one
experiment the paper cites. The handl stack is composed directly from
declared git deps (`src/lib.rs` assembles the kit — the same component
set catopt-kit composes; see the `[[gap]]` row in `manifest/kit.toml`
for why the direct `catopt-kit` edge is currently blocked upstream).
The `.hnl` artifact under measurement resolves in-tree — catopt-kit is
a workspace member of this repo (`packages/catopt-kit`); the
sibling-checkout / pkg-slot / fetched arms below keep a standalone
`research-mlopt` checkout working. Never vendored.

## `conv-bn-fold` — the founding result

Runs catopt's conv+BN weight-space fold end to end through the handl
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
workspace (see the `exclude` row in `Cargo.toml` — it floats on the
post-rename `hnl-engine-*` stack catopt-kit can't share a lockfile
with), so cargo invocations go through `--manifest-path`:

```bash
cargo run --release --manifest-path experiments/Cargo.toml --bin conv-bn-fold -- --repeat 200
cargo run --release --manifest-path experiments/Cargo.toml --bin conv-bn-fold -- --repeat 0   # certify only
cargo run --release --manifest-path experiments/Cargo.toml --bin conv-bn-fold -- --json       # machine-readable last line
```

## Resolving the artifact (no vendoring)

The program under measurement is catopt-kit's `hnl/conv_bn.hnl`. The
bin resolves it by this ladder, first hit wins:

1. `--conv-bn <path>` / `CONV_BN_HNL` — explicit override;
2. `packages/catopt-kit/hnl/conv_bn.hnl` — in-tree (this repo);
3. `$CATOPT_DIR/packages/catopt-kit/hnl/conv_bn.hnl` — a named catopt
   checkout (standalone arm);
4. the `.handl.toml` `[patch]` dir — `../catopt/packages/catopt-kit`
   sibling dev-link (inert where catopt isn't an org sibling);
5. the `.handl.toml` git-dep arm's package-cache slot —
   `$HANDL_PKG_DIR/catopt-kit` else `~/.handl/pkg/catopt-kit` (the dir
   the ADR-0010 fetch hook will populate; a checkout/symlink is today's
   stand-in);
6. `experiments/.cache/catopt-kit/hnl/conv_bn.hnl` — `make artifact`
   fetches it from the declared git source
   (`raw.githubusercontent.com/handl-lang/catopt/main/…`) when no local
   arm resolves.

The imported program's own `from "host-tensor" import tensor` resolves
through *this repo's* `experiments/.handl.toml` — the `.hnl`-level dep
manifest (`host-tensor` + `catopt-kit` declared by name, `[patch]` for
sibling checkouts).

## The Python-side evidence

`conv_fold_premise_pay_probe.py` (the gauntlet account — guard
transport, premise-anchored truth, typed pay, novelty) lives in catopt
and is *invoked*, not copied:

```bash
cd "$CATOPT_DIR" && .venv/bin/python lab/probes/conv_fold_premise_pay_probe.py
```
