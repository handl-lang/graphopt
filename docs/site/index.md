# graphopt — the research stack

<!-- provenance: graphopt:README.md@eea02f6 -->

**graphopt** is a categorical optimizer for neural-network computation
graphs — and a machine that writes its own rewrite rules, certifies
them, and learns to play the game they create. Programs, rules, and
derivations are all values in one calculus (n-morphisms), so a minted
rule re-enters the bank as a legal move and discovery closes into a
self-modifying game under a certifying referee.

- [How graphopt works](./explanation/mechanism) — syntax → structure →
  search → certificate.
- [Evaluation: the fourth dimension](./explanation/evaluation) —
  describe, rank, measure — never decide equivalence.

## Cross-cutting concepts

- [The three costs](./explanation/three-costs) — `dag_cost` guesses,
  fixtures measure, the verifier proves.
- [The game](./explanation/the-game) — boards, players, episodes, and
  the learned-policy loop.
- [The gauntlet](./explanation/the-gauntlet) — what a minted law must
  survive to ship.

## Learn by doing

The [tutorial track](./tutorials/) walks install → first `search()` →
first minted law → the scorecard. The contract surface is [the frozen
API reference](./reference/frozen-api); upstream, the repository's own
docs live at
[`handl-lang/graphopt/docs/`](https://github.com/handl-lang/graphopt/tree/main/docs).
