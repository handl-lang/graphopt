"""Train the learned mint proposer and emit the baked weight table.

Reads the overnight ledger cells
(``bench/results/checkpoints/overnight_*.jsonl`` — every
``("candidate", name)`` record carries its serialized law and its
gauntlet stages), fits the logistic ranker of
:mod:`mlopt_discovery.learned_propose` on the *survived-the-oracle*
label, and prints the trained table as the Python literal that bakes
into :data:`mlopt_discovery.learned_propose.DEFAULT_WEIGHTS` — the
same baker convention as ``train_meta_prior.py``.

The split is seeded and honest: a held-out slice is never trained on,
and the report prints the *enumeration baseline* beside the learned
numbers — the base oracle-pass rate (what a uniform proposer
achieves), the model's average precision, and precision@k, plus a
per-ledger breakdown (each file is a different mint arm, so the
per-arm rate shows where the ranker earns it).  A second, harder
evaluation holds out each ledger file in turn (leave-one-ledger-out)
— the cross-arm generalization number, since the mutation generator
mints shapes the enumeration arms never spell.

Usage::

    python scripts/train_learned_proposer.py
    python scripts/train_learned_proposer.py --epochs 400 --json /tmp/lp.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from mlopt_discovery import learned_propose as lp

#: Defaults — sized to the ~400-row ledger set.
EPOCHS = 400
LR = 0.3
L2 = 1e-3
SEED = 0
HELD_OUT = 0.25
#: Round the baked weights to this many decimals.
ROUND = 4


def _split(
    rows: list[lp.LabeledRow], held_out: float, seed: int
) -> tuple[list[lp.LabeledRow], list[lp.LabeledRow]]:
    """Seeded shuffle-split — ``held_out`` fraction never trains."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    cut = round(len(rows) * (1.0 - held_out))
    train_idx = set(idx[:cut])
    return (
        [r for i, r in enumerate(rows) if i in train_idx],
        [r for i, r in enumerate(rows) if i not in train_idx],
    )


def _by_ledger(
    model: lp.ProposerModel, rows: list[lp.LabeledRow]
) -> dict[str, dict[str, float]]:
    """Per-ledger-file metrics — each file is a different mint arm."""
    out: dict[str, list[lp.LabeledRow]] = {}
    for r in rows:
        out.setdefault(r.ledger, []).append(r)
    return {
        name: lp.evaluate(model, rs) for name, rs in sorted(out.items())
    }


def leave_one_ledger_out(
    rows: list[lp.LabeledRow],
    *,
    epochs: int = EPOCHS,
    lr: float = LR,
    l2: float = L2,
    seed: int = SEED,
) -> dict[str, dict[str, float]]:
    """Train on all but one ledger file; evaluate on the held-out arm.

    The cross-mint generalization number: each ledger is a different
    arm's candidate distribution, so a ranker that only memorized the
    enumeration families collapses here while one reading the
    pattern's shape still ranks.
    """
    ledgers = sorted({r.ledger for r in rows})
    out: dict[str, dict[str, float]] = {}
    for held in ledgers:
        train_rows = [r for r in rows if r.ledger != held]
        test_rows = [r for r in rows if r.ledger == held]
        model = lp.train(
            train_rows, epochs=epochs, lr=lr, l2=l2, seed=seed
        )
        out[held] = lp.evaluate(model, test_rows)
    return out


def baked_table(model: lp.ProposerModel) -> dict[str, float]:
    """Return the snapshot — rounded, nonzero weights only."""
    return {
        name: round(w, ROUND)
        for name, w in sorted(model.weights.items())
        if abs(w) > 10.0**-ROUND
    }


def main(argv: Any = None) -> int:
    """Train, measure the split, print the bake."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ledgers",
        nargs="*",
        default=None,
        help="ledger JSONL paths (default: the overnight checkpoint glob)",
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--l2", type=float, default=L2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--held-out", type=float, default=HELD_OUT)
    ap.add_argument("--json", default=None, help="write metrics JSON")
    ap.add_argument(
        "--lolo",
        action="store_true",
        help="also run the leave-one-ledger-out cross-arm eval",
    )
    args = ap.parse_args(argv)

    paths = (
        [Path(p) for p in args.ledgers]
        if args.ledgers
        else lp.expand_ledgers()
    )
    rows = lp.load_ledger_rows(paths)
    print(f"ledgers: {[p.name for p in paths]}")
    print(
        f"rows: {len(rows)}  positives: {sum(1 for r in rows if r.y)}"
    )

    train_rows, test_rows = _split(rows, args.held_out, args.seed)
    model = lp.train(
        train_rows,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )
    report: dict[str, Any] = {
        "n_rows": len(rows),
        "n_pos": sum(1 for r in rows if r.y),
        "train": lp.evaluate(model, train_rows),
        "held_out": lp.evaluate(model, test_rows),
        "held_out_by_ledger": _by_ledger(model, test_rows),
    }
    if args.lolo:
        report["lolo"] = leave_one_ledger_out(
            rows,
            epochs=args.epochs,
            lr=args.lr,
            l2=args.l2,
            seed=args.seed,
        )

    print(
        f"train     n={report['train']['n']:.0f} "
        f"base={report['train']['base_rate']:.3f} "
        f"ap={report['train']['ap']:.3f}"
    )
    ho = report["held_out"]
    print(
        f"held-out  n={ho['n']:.0f} base={ho['base_rate']:.3f} "
        f"ap={ho['ap']:.3f} p@10={ho['p@10']:.3f} p@25={ho['p@25']:.3f}"
    )
    for name, m in report["held_out_by_ledger"].items():
        print(
            f"  {name:36s} n={m['n']:.0f} base={m['base_rate']:.3f} "
            f"ap={m['ap']:.3f}"
        )
    if args.lolo:
        print("leave-one-ledger-out (train on the other arms):")
        for name, m in report["lolo"].items():
            print(
                f"  {name:36s} n={m['n']:.0f} base={m['base_rate']:.3f} "
                f"ap={m['ap']:.3f} p@10={m['p@10']:.3f}"
            )

    table = baked_table(model)
    print(
        f"\n# {len(table)} weights — bake into "
        "learned_propose.DEFAULT_WEIGHTS:"
    )
    print(f"DEFAULT_WEIGHTS = {table!r}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
