"""The ``play`` CLI — ``python -m mlopt_discovery.play``.

Trains the learned arm on one case stream, evals on another, prints
the paired-boards table; ``--deliver`` / ``--probe`` / ``--zoo`` /
``--scan-probe`` / ``--decay-probe`` run the report arms, and
``--domain-spec`` / ``--player`` extend the registry.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from gamekit import engine
from gamekit.players import LinearPolicy

from .gen import _zoo_rows, gen_probe
from .players import _random_policy
from .probes import decay_probe, scan_probe
from .registry import DOMAINS, Domain, deliver


def _pkg_doc() -> str:
    """Return the package docstring — the CLI ``--help`` text."""
    from mlopt_discovery import play as _p

    return _p.__doc__ or ""


def _arms(domain: Domain, player: LinearPolicy, seed: int) -> dict:
    """Build the arm set for one domain: baselines + learned pair."""
    rnd = _random_policy(domain.legal)
    arms: dict[str, Any] = {
        "random": lambda i: rnd(seed + 10_000 + i),
        "learned": lambda i: player.frozen(seed + 20_000 + i),
        "learned-greedy": lambda i: player.frozen(
            seed + 30_000 + i, greedy=True
        ),
    }
    if domain.scripted is not None:
        arms = {"scripted": lambda i: domain.scripted(), **arms}
    return arms


def _table(table: dict[str, list[dict]]) -> str:
    """Render per-arm means over eval rows — generic edition."""
    head = f"{'player':<15} {'cases':>6} {'reward':>9} {'cost':>7} {'cert':>5} {'steps':>6}"
    lines = [head, "-" * len(head)]
    for name, rows in table.items():
        n = max(len(rows), 1)
        costs = [r["cost"] for r in rows if r["cost"] is not None]
        certs = sum(1 for r in rows if r["cert"])
        lines.append(
            f"{name:<15} {len(rows):>6} "
            f"{sum(r['reward'] for r in rows) / n:>9.2f} "
            f"{(sum(costs) / len(costs)) if costs else float('nan'):>7.2f} "
            f"{certs:>5} "
            f"{sum(r['steps'] for r in rows) / n:>6.1f}"
        )
    return "\n".join(lines)


def _extras(args: Any, d: Any, ev: list, fallback: Any) -> None:
    """Run the ``--*probe`` / ``--deliver`` / ``--zoo`` report arms."""
    if args.deliver:
        arm = d.scripted() if d.scripted is not None else fallback
        res = deliver(args.domain, ev[0], arm, budget=args.budget)
        shown = {k: v for k, v in res.items() if k != "module"}
        print(f"deliver[{ev[0][0]}]: {shown}")  # stdout-compat
    if args.probe:
        for case in ev:
            if DOMAINS[args.domain] is DOMAINS["gen"]:
                print(  # stdout-compat
                    f"probe[{case[0]}]: "
                    f"{gen_probe(case, pay_gate=getattr(args, 'pay_gate', False))}"
                )
    if args.zoo:
        _zoo_rows(args)
    _kernel_probes(args)


def _kernel_probes(args: Any) -> None:
    """Run the generated-kernel timing arms (CUDA + triton)."""
    if args.scan_probe:
        for row in scan_probe():
            print(  # stdout-compat
                f"scan[{row.get('shape', '-')}]: {row}"
            )
    if args.decay_probe:
        for row in decay_probe():
            print(  # stdout-compat
                f"decay[{row.get('shape', '-')}]: {row}"
            )


def main(argv: list[str] | None = None) -> int:
    """Train the learned arm on one case stream, eval on another."""
    ap = argparse.ArgumentParser(
        # ``prog`` pins the usage line: as a package, ``-m`` would
        # otherwise report ``__main__.py`` where the module reported
        # ``play.py``.
        prog="play.py",
        description=_pkg_doc(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--domain",
        default="meta",
        help=(
            "a registered domain name — built-ins plus any seated "
            "via --domain-spec or a downstream register() plugin "
            "(kernelopt's 'kernel')"
        ),
    )
    ap.add_argument(
        "--domain-spec",
        action="append",
        default=[],
        metavar="PATH|NAME",
        help=(
            "load a domain-spec file (JSON path or packaged "
            "domains/<name>.json) and register its domain before "
            "play — repeatable (mlopt_discovery.domain_spec)"
        ),
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cases", type=int, default=40)
    ap.add_argument("--eval-cases", type=int, default=10)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument(
        "--deliver",
        action="store_true",
        help="lower + verify the first eval case's extraction",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="run the gen domain's claim→deliver→time probe per eval case",
    )
    ap.add_argument(
        "--zoo",
        action="store_true",
        help="run the claim→deliver→time probe over the held-out zoo",
    )
    ap.add_argument(
        "--zoo-d",
        type=int,
        default=16,
        help="zoo model width (default 16 — the honest small probe)",
    )
    ap.add_argument(
        "--zoo-t",
        type=int,
        default=8,
        help="zoo sequence length (default 8 — kernels pay ≥64)",
    )
    ap.add_argument(
        "--zoo-b",
        type=int,
        default=2,
        help="zoo batch size (default 2 — the honest small probe)",
    )
    ap.add_argument(
        "--pay-gate",
        action="store_true",
        help=(
            "route --probe/--zoo workloads through the pay predictor "
            "(mlopt_discovery.pay_gate) and skip the elementwise-bound "
            "signature before mint/referee/search spend — scheduling "
            "only, default off"
        ),
    )
    ap.add_argument(
        "--scan-probe",
        action="store_true",
        help=(
            "time the generated Triton scan kernel against the "
            "spelled loop and the carrier executor path (CUDA)"
        ),
    )
    ap.add_argument(
        "--decay-probe",
        action="store_true",
        help=(
            "time the generated fused (s·d)@v Triton kernel against "
            "eager, torch.compile, and the om_decayed executor (CUDA)"
        ),
    )
    ap.add_argument(
        "--player",
        default=None,
        metavar="NAME",
        help=(
            "evaluate a registered players/ module by name instead of "
            "training the learned arm — eval-only, paired against "
            "``random`` (see players.registry.names())"
        ),
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    for spec_arg in args.domain_spec:
        from .. import domain_spec

        # ``domains=DOMAINS``: under ``python -m mlopt_discovery.play``
        # this module *is* ``__main__`` — a different object than the
        # ``mlopt_discovery.play`` domain_spec would import, so the
        # registry must be passed explicitly.
        domain_spec.register_file(spec_arg, domains=DOMAINS)
    if args.domain not in DOMAINS:
        print(  # stdout-compat
            f"unknown domain {args.domain!r} — known: "
            f"{sorted(DOMAINS)!r}"
        )
        return 2
    d = DOMAINS[args.domain]
    train = d.cases(args.seed, args.train_cases)
    ev = d.cases(args.seed + 7919, args.eval_cases)
    if args.player is not None:
        from gamekit.players import registry

        try:
            mk = registry.make(
                args.player, args.domain, seed=args.seed, ctx={}
            )
        except ValueError as e:
            print(f"--player: {e}")  # stdout-compat
            return 2
        rnd = _random_policy(d.legal)
        arms = {
            args.player: mk,
            "random": lambda i: rnd(args.seed + 10_000 + i),
        }
        fallback = mk(0)
        totals: list = []
    else:
        player = LinearPolicy(
            seed=args.seed,
            legal=d.legal,
            featurizer=d.featurizer,
            features=d.features,
            temperature=args.temperature,
            epsilon=args.epsilon,
        )
        totals = engine.train_policy(
            player, train, d.board_of, budget=args.budget
        )
        arms = _arms(d, player, args.seed)
        fallback = player.frozen(0)
    table = engine.evaluate(arms, ev, d.board_of, budget=args.budget)
    print(f"== play — domain {args.domain} ==")  # stdout-compat
    print(_table(table))  # stdout-compat
    if args.player is not None:
        print(  # stdout-compat
            f"eval-only — registered player {args.player!r}"
        )
    else:
        decade = max(len(totals) // 10, 1)
        means = [
            round(sum(totals[i : i + decade]) / decade, 2)
            for i in range(0, len(totals) - decade + 1, decade)
        ]
        print(f"train decade means: {means}")  # stdout-compat
    _extras(args, d, ev, fallback)
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(
                {"table": table, "train_totals": totals}, f, indent=1
            )
    return 0
