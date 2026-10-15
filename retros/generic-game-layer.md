# Retro — the generic game layer (engine + players + play)

The codebase had grown three boards and two learned players with
duplicated drivers.  The productization pass factors the shared
shape out:

- `catopt_discovery.engine` — the `Board` protocol
  (`observe() -> state` carrying `.done`, `step(action) ->
  (state, report)` carrying `.reward`/`.terminal`), the generic
  `run_episode`, `train_policy` (reward-to-go plumbing), and
  `evaluate` (paired-board arm comparison).  The two arena
  `run_episode` functions were already copies; now there is one.
- `catopt_discovery.players` — `LinearPolicy`: the softmax-linear
  REINFORCE learner with the domain injected (`legal=`,
  `featurizer=`, `features=`).  `MetaLearnedPlayer` survives as a
  three-line preset binding the meta schema — same tests, same
  behavior, no fork of the machinery.
- `catopt_discovery.play` — the registry + CLI.  A domain is data:
  `cases` (seeded corpus), `board_of` (case -> Board), `legal`,
  `featurizer`, optional `scripted` arm.  Three shipped:

  - `meta` — the meta-arena over `meta_player.gen_cases`;
  - `joint` — forward + derived gradients under one `joint` root
    on the meta-arena board (shape-varied archetypes: silu /
    softmax-spine / linear+tanh — the reverse handler's programs
    are board citizens, which is the point);
  - `search` — `catopt_core.search_env.SearchEnv` under the Board
    contract (`SearchBoard`): rule-name actions, env reward =
    normalized cost improvement, horizon/patience terminator;
  - `torch` — real `nn.Module`s (`Sequential` MLPs at seeded
    widths) exported via `TorchSource`, played under the real
    `TorchSink().supported_ops` bound: the first product-surface
    board — scripted pays **+0.23** mean reward on the demo
    corpus (the sink's fold laws find real wins, e.g. SiLU folds
    on `Linear→SiLU` blocks), where the synthetic boards mostly
    tie at baseline.

```
python -m catopt_discovery.play --domain meta|joint|search|torch \
    --train-cases N --eval-cases M --budget B
```

## What it changes

Adding a domain used to mean writing a board *and* a player.  Now
a domain is a registration entry; the learned arm works on any
board whose featurizer exists — verified by driving a toy board
through `LinearPolicy` with a two-feature schema
(`test_generic_domain_plays`).

The `search` domain is the honest weak link: rule-fire steps on
the current toy corpus find no improvement (all-zero reward table)
— the env works, the corpus is thin; richer search corpora are a
data problem, not a machinery one.

## Where this goes

The product claim this sets up: *a verified-optimization gym*.
Every domain is "a program + a legal move set + a referee"; the
reward is certified improvement.  Next: a `Board` for the real
torch pipeline (`TorchSource` -> optimize episode -> `TorchSink`
runnable out), which is the user-facing "play my model" surface.
