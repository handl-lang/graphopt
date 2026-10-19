# manifest/ — the machinery, as data

`kit.toml` is the paper's composition claim made checkable. Every
component the research composes is a declared row — repo, float policy
(`branch = "main"`, no rev pins), the crates/packages consumed, and the
role it plays in the lift → discover+admit → lower arc. The Rust arm is
*enforced*: `experiments/Cargo.toml`'s dep edges are these rows, so a
drift between the manifest and the build is a visible diff, not a
silent fork.

## Schema (informal, `handl-lang/research-kit@1`)

| section | meaning |
|---|---|
| `[meta]` | repo name, paper path, the claim, the founding result |
| `[[component]]` | one composed component: `name`, `repo`, `float`, `kind` (`cargo-git` / `python-git`), `path` (for monorepo members), `crates`/`packages`, `role` in the arc |
| `[[experiment]]` | one runnable evidence leg: `run` command, what it `produces`, the artifact it measures |
| `[[gap]]` | a surfaced gap — what, where it lives upstream, the honest workaround (never vendoring) |

## The two arms

- **Rust arm** (`kind = "cargo-git"`) — resolved by Cargo from the git
  sources, `branch = "main"`. Locally the deps land on sibling
  checkouts via `source ~/.cache/devin/handl_deps.env` (git
  `insteadOf`) or the `[patch]` tables `scripts/dev-link.sh`
  regenerates. This is the arm `make experiment` exercises.
- **Python arm** (`kind = "python-git"`) — mlopt's discovery engine.
  Python has no workspace-level git-dep equivalent in this repo, so
  the arm is *declared* (consumed as `git+https://…` deps inside a
  mlopt venv) and its probes are *invoked* in a mlopt checkout, not
  copied here.

## Why a manifest and not just Cargo.toml

The paper's claim is that the machinery is *data* — the composition is
the contribution. `Cargo.toml` proves the Rust half builds; the
manifest names the whole stack (including the Python arm and the
experiments themselves) in one diff-able artifact, so the research
story is reproducible from declared deps rather than from a monorepo's
accidental adjacency.
