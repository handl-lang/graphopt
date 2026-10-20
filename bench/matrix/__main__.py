"""``python -m bench.matrix`` — the systematic-eval matrix CLI.

    .venv/bin/python -m bench.matrix --smoke                # seconds
    .venv/bin/python -m bench.matrix --players playbook,linear \
        --referees naive,full --boards open,search
    .venv/bin/python -m bench.matrix --coverage            # audit only

Writes ``<out>/matrix.md`` + ``<out>/matrix.json`` — default
``docs/`` (the committed, versioned matrix).  Cells are the designed
subset: every requested player that plays each board, refereed by
every requested arm — ``contraction``/``player`` produce no equality
claims and are marked ``unrun`` with the reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlopt_discovery.play import DOMAINS

from bench.benchkit import checkpoint as checkpoint_mod
from bench.matrix import table
from bench.matrix.runner import coverage, run_matrix
from bench.referees import registry as refs

#: Boards the matrix can drive.
ALL_BOARDS = (*sorted(DOMAINS), "contraction")

SMOKE = {
    "meta_cases": 3,
    "meta_budget": 6,
    "open_pages": 2,
    "open_page": 6,
    "open_budget": 7,
}


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the matrix, emit the table."""
    p = argparse.ArgumentParser(prog="bench.matrix")
    p.add_argument(
        "--players",
        default="playbook,linear,random",
        help="comma-separated player names",
    )
    p.add_argument(
        "--referees",
        default=",".join(refs.names()),
        help="comma-separated referee arms",
    )
    p.add_argument(
        "--boards",
        default="open,search",
        help=f"comma-separated boards (known: {', '.join(ALL_BOARDS)})",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--meta-cases", type=int, default=8)
    p.add_argument("--meta-budget", type=int, default=10)
    p.add_argument("--open-pages", type=int, default=4)
    p.add_argument("--open-page", type=int, default=8)
    p.add_argument("--open-budget", type=int, default=9)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="shrink every knob — the seconds-scale lane",
    )
    p.add_argument(
        "--coverage",
        action="store_true",
        help="print the cell-space audit and exit",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("docs"),
        help="output directory (default docs/)",
    )
    p.add_argument(
        "--pin",
        action="store_true",
        help="pin the table as bench/baselines/matrix.json",
    )
    p.add_argument(
        "--baselines",
        type=Path,
        default=Path("bench/baselines"),
    )
    p.add_argument(
        "--no-ledger",
        action="store_true",
        help="skip the ledger append (bench/results/ledger.jsonl)",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const=str(checkpoint_mod.DEFAULT_DIR / "matrix.jsonl"),
        default=None,
        metavar="PATH",
        help="resume from the cell checkpoint "
        "(default: bench/results/checkpoints/matrix.jsonl); "
        "recorded cells are reused, a crash costs a cell",
    )
    args = p.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)
    if args.smoke:
        for key, val in SMOKE.items():
            setattr(args, key, val)

    players = [x.strip() for x in args.players.split(",") if x.strip()]
    referees = [
        x.strip() for x in args.referees.split(",") if x.strip()
    ]
    boards = [x.strip() for x in args.boards.split(",") if x.strip()]
    unknown = [b for b in boards if b not in ALL_BOARDS]
    if unknown:
        print(
            f"unknown board(s): {', '.join(unknown)} — "
            f"known: {', '.join(ALL_BOARDS)}"
        )
        return 2
    unknown_r = [r for r in referees if r not in refs.names()]
    if unknown_r:
        print(
            f"unknown referee(s): {', '.join(unknown_r)} — "
            f"registered: {', '.join(refs.names())}"
        )
        return 2
    from gamekit.players import registry as players_reg

    try:
        for name in players:
            players_reg.spec(name)
    except ValueError as e:
        print(e)
        return 2

    cov = coverage(boards, players, referees)
    if args.coverage:
        for c in cov:
            mark = "run " if c.run else "SKIP"
            print(
                f"  {mark} {c.board:<12} {c.player:<14} "
                f"{c.referee:<8} {c.reason}"
            )
        return 0

    protocol = {
        "seed": args.seed,
        "meta_cases": args.meta_cases,
        "meta_budget": args.meta_budget,
        "meta_held_out": "gen_cases(seed+7919)",
        "open_pages": args.open_pages,
        "open_page": args.open_page,
        "open_budget": args.open_budget,
        "oracle_pairing": "memoized per-candidate-seeded verdicts",
    }
    ckpt = None
    if args.resume is not None:
        manifest = checkpoint_mod.new_manifest(
            "matrix",
            {
                **protocol,
                "players": players,
                "referees": referees,
                "boards": boards,
            },
            device=args.device,
        )
        ckpt = checkpoint_mod.Checkpoint.open(args.resume, manifest)
        print(
            f"resume: {len(ckpt.entries)} cells already done "
            f"({args.resume})",
            flush=True,
        )
    cells = run_matrix(
        boards,
        players,
        referees,
        seed=args.seed,
        meta_cases=args.meta_cases,
        meta_budget=args.meta_budget,
        open_pages=args.open_pages,
        open_page=args.open_page,
        open_budget=args.open_budget,
        checkpoint=ckpt,
    )
    table.write_outputs(
        cells,
        coverage=cov,
        md_path=args.out / "matrix.md",
        json_path=args.out / "matrix.json",
        protocol=protocol,
        device=args.device,
        seed=args.seed,
    )
    from bench.benchkit import table as bktable

    record = bktable.table_record(
        "matrix",
        (
            {
                "name": f"{c.board}/{c.player}/{c.referee}",
                "metrics": {
                    "admit_rate": c.admit_rate,
                    "value": c.value,
                },
            }
            for c in cells
            if c.run
        ),
        directions={"admit_rate": "down", "value": "up"},
        findings=[
            {
                "claim": "player claims survive the referee dial",
                "verdict": "parity",
                "headline": (
                    f"{sum(c.claims for c in cells)} claims, "
                    f"{sum(c.admitted for c in cells)} admitted "
                    f"over {len(cells)} cells"
                ),
            }
        ],
        provenance={"quick": bool(args.smoke)},
        seed=args.seed,
        device=args.device,
    )
    if not args.no_ledger:
        bktable.append_record(record)
    if args.pin:
        path = bktable.write_table_baseline(record, args.baselines)
        print(f"  -> pinned {path}")
    print(
        f"matrix: {len(cells)} cells, "
        f"{sum(not c.run for c in cov)} unrun",
        flush=True,
    )
    print(f"  -> {args.out / 'matrix.md'}")
    print(f"  -> {args.out / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
