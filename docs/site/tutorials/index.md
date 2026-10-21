# The research track

<!-- provenance: graphopt:README.md@eea02f6 + AGENTS.md -->

::: info Rename note
The research stack lives in the **`graphopt`** repository — renamed
from `research-mlopt` for the repo split, which has landed: the `eqsat`
substrate and the `gamekit` game layer are their own repositories now,
pinned here as git deps (import names unchanged). The `research/` paths
in this section already swept their provenance to `graphopt:` — the
rename was a sweep, not a rewrite.
:::

This track is the learn-by-doing path for **mlopt** — the categorical
optimizer + law-discovery engine in
[`handl-lang/graphopt`](https://github.com/handl-lang/graphopt):
a model graph goes in, equality saturation explores its equivalence
class, and a cheaper, *certified* form comes out — while a separate
loop mints new rewrite laws and admits them through a measured
gauntlet.

Work through the steps in order; each produces a real artifact in the
repo:

1. [Set up graphopt](./setup) — clone, `uv sync`, verify the dev environment.
2. [Your first `search()`](./first-search) — a model in, a cheaper certified form out.
3. [Watch a law get minted](./mint-a-law) — the discovery loop and the admission gauntlet.
4. [Run the scorecard](./run-the-scorecard) — regenerate the named-optimization table and read a row.

## The concepts behind the steps

- [How mlopt works](../explanation/mechanism) — syntax → structure → search → certificate.
- [The three costs](../explanation/three-costs) — how the engine knows a form is *better*.
- [The game](../explanation/the-game) — boards, players, and the learned-policy loop.
- [The gauntlet](../explanation/the-gauntlet) — what a minted law must survive to ship.

## The interactive course (runs locally)

[`docs/tutorials/course/`](https://github.com/handl-lang/graphopt/tree/main/docs/tutorials/course)
in the repo is a **28-level interactive course** on mlopt's
architecture — one idea per level, a worked example, an interactive,
and a gated checkpoint per level. It is a Vite+ app with its own
toolchain, **not published on this site**; it builds standalone and
runs locally:

```sh
cd graphopt/docs/tutorials/course
vp install                 # install the workspace (pnpm + vite-plus)
vp -C apps/learn dev       # http://localhost:5173
vp run -r build            # engine → dist, app → dist
```

## When to use another section

- For the frozen contract surface, see [the frozen API reference](../reference/frozen-api).
- For the language platform track, see [Handl Language tutorials](../../handl/tutorials/).
