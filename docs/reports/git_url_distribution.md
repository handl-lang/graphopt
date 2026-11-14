# Git-URL tagged distribution — survey + scheme

*Status: scheme adopted, tags created locally, pins pending push.*
*Survey of `~/GitHub/handl-lang/*` at the commits in §3 (2026-11-13).*

## TL;DR

- Distribution is **git-URL at a tag**, never PyPI: users install with
  `git+https://github.com/handl-lang/<repo>@vX.Y.Z` (or `git+ssh`).
- Tag format: **annotated semver `vX.Y.Z` per repo** (`v0.1.0` first).
  uv git sources pin a *repo-level* rev, so one tag per repo per
  release; `subdirectory = "packages/<pkg>"` selects the package inside.
- Tags `v0.1.0` were created **locally** on the seven consumed repos
  (listed below). **Nothing was pushed** — the parent pushes.
- `[tool.uv.sources]` pins are **deliberately left on `branch = "main"`**:
  verified that `uv` cannot resolve a tag that does not exist on the
  remote (`fatal: couldn't find remote ref refs/tags/v0.1.0`), so
  flipping pins before the push would break `uv sync`/`uv.lock` for
  everyone. The exact pending edits are in §4 — they land in the same
  commit that pushes the tags, followed by `uv lock`.

## 1. Survey — who consumes whom

### Python / uv chain (the live graph)

| consumer | dep | source repo | subdir | current pin |
|---|---|---|---|---|
| graphopt | `mlopt-core`, `mlopt-orchestrator` | `handl-lang/eqsat` | `packages/mlopt-core`, `packages/mlopt-orchestrator` | `branch = "main"` |
| graphopt | `gamekit`, `benchkit` | `handl-lang/gamekit` | `packages/gamekit`, `packages/benchkit` | `branch = "main"` |
| graphopt | `mlopt-torch`, `mlopt-carriers`, `mlopt-cuda`, `mlopt-discovery`, `mlopt-onnx` | workspace members (`packages/*`) | — | `{ workspace = true }` |
| gamekit | `mlopt-core` (benchkit only) | `handl-lang/eqsat` | `packages/mlopt-core` | `branch = "main"` |
| kernelopt | `mlopt-core`, `mlopt-orchestrator` | `handl-lang/eqsat` | same | `branch = "main"` |
| kernelopt | `mlopt-torch`, `mlopt-carriers`, `mlopt-discovery` | `handl-lang/graphopt` | `packages/*` | `branch = "main"` |
| kernelopt | `gamekit`, `benchkit` | `handl-lang/gamekit` | same | `branch = "main"` |
| kernelopt | `lib-kernel-ir` | sibling checkout | — | `path = "../lib-kernel-ir"` |
| kit-kernel-ir | `lib-kernel-ir`, `lib-triton-qgemv` | sibling checkouts | — | `path = "../…"` |

Notes:

- All git URLs are `https://github.com/handl-lang/<repo>.git` — the
  pyproject *comments* say "git+ssh" but the declared scheme is HTTPS
  (works for public repos without SSH keys; `host/dagger` is the only
  other `tool.uv.sources` user and it is all `path`/`workspace`).
- `lib-kernel-ir`, `kit-kernel-ir`, `lib-triton-qgemv` have
  `pyproject.toml` but are consumed **only as path deps** — they are
  not yet git-distributed (kernelopt's pytest `pythonpath` also reads
  `../lib-kernel-ir/src` directly). Tags exist so a future
  `git = …, subdirectory = …` pin is possible; switching them to git
  deps is a separate decision (they'd need `src/`-layout packaging
  verification — `lib-kernel-ir` already ships `src/`).
- `host-tensor`, `lib-tensor` have **no pyproject** — Rust/cdylib-side
  only, see §5.
- kernelopt is **PRIVATE** on GitHub (`gh repo view` → `"visibility":"PRIVATE"`);
  graphopt/eqsat/gamekit are PUBLIC. kernelopt's own tag is created but
  its git-URL distribution waits on the public flip.

### Rust / kit chain (~108 `Cargo.toml` files)

Every `git = "https://github.com/handl-lang/<repo>"` dep floats on
`branch = "main"` day-to-day; `Cargo.lock` records the resolved sha.
At release, `host/scripts/pin-git-revs.py` rewrites each selector to
`rev = "<HEAD sha>"` (immutable; `--check` reports drift) — this is the
already-documented policy in `host/AGENTS.md` ("float on `main`, pin
at release"). Cargo *does* accept `tag = "…"` selectors, so the semver
tags created here can serve the Rust chain too, but the existing
sha-pin tooling is the incumbent scheme — tags are additive, not a
replacement. graphopt's own `packages/mlopt-kit` +
`packages/mlopt-experiments` Cargo.tomls depend on ~15 handl repos
(kernel, host, engine-*, kit-prelude-lang, host-tensor, egraph,
discovery, graphopt itself) all on `branch = "main"`.

### Non-Python org

~150 sibling repos are kit/handl libraries (`kit.toml`/`bupha.toml`
manifests, `.hnl` sources, Rust crates). They are consumed through the
kit/`.handl.toml` machinery (platform's fetch-on-miss resolver clones
the declared repo@selector into `~/.handl/pkg/<name>` — see
`graphopt/manifest/kit.toml` `[[gap]]` entries) and through the Cargo
git-dep web above. None have PyPI-style packaging; the same
`vX.Y.Z`-per-repo tag convention applies if/when they are released.

## 2. Tag scheme

- **Format:** annotated tag `vMAJOR.MINOR.PATCH` on the repo's
  `main` HEAD at release time. First tag everywhere: **`v0.1.0`**.
- **Granularity:** one tag per *repo* (uv/cargo git selectors resolve a
  repo rev; `subdirectory`/crate-path picks the package). No per-package
  tags — they would lie about what a rev contains.
- **Consumers pin:** `tag = "v0.1.0"` in `[tool.uv.sources]` /
  `[dependencies]`, or end users write
  `pkg @ git+https://github.com/handl-lang/<repo>@v0.1.0` — for
  subdirectory packages uv/pip need the `#subdirectory=` fragment or a
  `[tool.uv.sources]` entry in the *consumer's* pyproject.
- **Bumping:** tag the new `main` HEAD `v0.2.0` (or `v0.1.1` for
  fixes), update pins, `uv lock`. Never move an existing tag.

## 3. Tags created (LOCAL ONLY — not pushed)

| repo | tag | commit | visibility |
|---|---|---|---|
| eqsat | `v0.1.0` | `38422b3` | public |
| gamekit | `v0.1.0` | `2f0cee2` | public |
| graphopt | `v0.1.0` | `4c7c03b` | public |
| kernelopt | `v0.1.0` | `a31dc26` | **private** |
| lib-kernel-ir | `v0.1.0` | `7bd7fbe` | — |
| kit-kernel-ir | `v0.1.0` | `262fb88` | — |
| lib-triton-qgemv | `v0.1.0` | `1613db3` | — |

All point at `main` HEAD == `origin/main` (verified `rev-list` 0/0
ahead/behind; graphopt's 2 dirty files and host-tensor/lib-tensor's
dirty trees are uncommitted and therefore not in any tag).

**To publish:** `git -C <repo> push origin v0.1.0` for each row.

## 4. Pending pin changes (land AFTER the tag push)

`uv` resolves git tags against the **remote** — a local-only tag is
invisible (`uv lock` fails with `couldn't find remote ref`). Verified
empirically in a scratch project. Order of operations:

1. Push the tags (§3).
2. Apply these edits, then `uv lock` in each repo.

**graphopt `pyproject.toml` `[tool.uv.sources]`:**

```diff
-mlopt-core = { git = "https://github.com/handl-lang/eqsat.git", subdirectory = "packages/mlopt-core", branch = "main" }
-mlopt-orchestrator = { git = "https://github.com/handl-lang/eqsat.git", subdirectory = "packages/mlopt-orchestrator", branch = "main" }
-gamekit = { git = "https://github.com/handl-lang/gamekit.git", subdirectory = "packages/gamekit", branch = "main" }
-benchkit = { git = "https://github.com/handl-lang/gamekit.git", subdirectory = "packages/benchkit", branch = "main" }
+mlopt-core = { git = "https://github.com/handl-lang/eqsat.git", subdirectory = "packages/mlopt-core", tag = "v0.1.0" }
+mlopt-orchestrator = { git = "https://github.com/handl-lang/eqsat.git", subdirectory = "packages/mlopt-orchestrator", tag = "v0.1.0" }
+gamekit = { git = "https://github.com/handl-lang/gamekit.git", subdirectory = "packages/gamekit", tag = "v0.1.0" }
+benchkit = { git = "https://github.com/handl-lang/gamekit.git", subdirectory = "packages/benchkit", tag = "v0.1.0" }
```

**gamekit `pyproject.toml`:** same `branch = "main"` → `tag = "v0.1.0"`
on the `mlopt-core` eqsat source.

**kernelopt `pyproject.toml`:** `tag = "v0.1.0"` on all seven git
sources (eqsat ×2, graphopt ×3, gamekit ×2). `lib-kernel-ir` stays a
`path` dep until it is deliberately converted to a git dep.

**Rust chain:** unchanged — keep `branch = "main"` + the
`pin-git-revs.py` release flow; optionally adopt `tag = "vX.Y.Z"`
selectors once the org tags releases there too.

## 5. Honest limits

- **kernelopt is private.** Its `v0.1.0` tag is local-only until the
  repo flips public; git-URL installs of kernelopt fail for outsiders
  regardless of pin style until then. Do not flip visibility without
  the owner's go.
- **`lib-kernel-ir` / `kit-kernel-ir` / `lib-triton-qgemv`** are path
  deps, not git deps — tags exist but nothing resolves them over git
  yet. `kit-kernel-ir`'s pytest `pythonpath` also reaches into
  `../lib-kernel-ir/src`, so a git-dep conversion needs packaging work.
- **graphopt's Rust crates** (`mlopt-kit`, `mlopt-experiments`) drag
  in ~15 handl repos as `branch = "main"` Cargo deps — reproducible
  only via `Cargo.lock` sha + `pin-git-revs.py`, not tags (yet).
- **`host-tensor` / `lib-tensor`** have uncommitted changes and no
  Python packaging — out of scope for the uv chain.
- **Wheel installs are not self-contained** (documented in
  `docs/reports/phase_a_packaging.md`): a `mlopt-torch` wheel still
  needs `mlopt-core`/`gamekit` from their repos — git-URL deps are the
  install path, wheels are artifacts.
- **Tags are immutable by convention** — never move `v0.1.0`; cut
  `v0.1.1`/`v0.2.0` instead.

## 6. Docs touched

- `docs/site/tutorials/setup.md` — install callout now names the tag
  scheme + git-URL form (and corrects the stale "git+ssh" wording —
  the declared sources are HTTPS).
- `README.md` — "Install" note under Reproduce.
- `eqsat/README.md`, `gamekit/README.md`, `kernelopt/README.md` —
  dep-consumption notes now name the `v0.1.0` tag.
- `pyproject.toml` comments (graphopt/gamekit/kernelopt) — record the
  pending `branch → tag` flip so the next editor sees it.

## Verification

- `uv lock --check` in graphopt: **clean** (213 packages resolved) —
  pins unchanged, `uv.lock` untouched.
- `uv lock` against `tag = "v0.1.0"` pre-push: **fails** as expected
  (`couldn't find remote ref`) — the reason pins wait for the push.
