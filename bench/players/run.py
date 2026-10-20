"""``python -m bench.players.run`` — the player leaderboard CLI.

    .venv/bin/python -m bench.players.run --smoke          # seconds
    .venv/bin/python -m bench.players.run                  # full protocol
    .venv/bin/python -m bench.players.run --players linear,playbook \
        --boards meta,open

Writes ``<out>/leaderboard_players.md`` + ``<out>/leaderboard_players
.json`` — default ``docs/`` (the committed, versioned leaderboard).
The protocol is ``bench/drivers/policy/policy_baseline.py``'s: paired boards, held-out
case streams, and on ``open`` memoized per-candidate oracle verdicts
so every arm sees literally identical draws.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlopt_discovery.play import DOMAINS
from mlopt_discovery.players import registry

from bench.players.leaderboard import write_outputs
from bench.players.runner import run_board

#: Default sizes — the ``policy_baseline`` protocol where one exists.
SEED = 0
META_CASES = 16
META_BUDGET = 12
OPEN_PAGES = 36  # odd pages evaluate; even pages feed neural's train
OPEN_PAGE = 16
OPEN_BUDGET = OPEN_PAGE + 1
NEURAL_EPOCHS = 300
NEURAL_HIDDEN = 32
CONTRACTION_SCALES = "8,12"
CONTRACTION_INSTANCES = 4
CONTRACTION_SAMPLES = 32

#: ``--smoke`` — every knob small enough for a seconds-scale lane.
SMOKE = {
    "meta_cases": 4,
    "meta_budget": 8,
    "open_pages": 2,
    "open_page": 8,
    "open_budget": 9,
    "neural_epochs": 20,
    "contraction_scales": "8",
    "contraction_instances": 2,
    "contraction_samples": 8,
}

#: Boards the leaderboard can drive — every play domain plus the
#: order-family contraction board.
ALL_BOARDS = (*sorted(DOMAINS), "contraction")


def _print_row(row) -> None:
    """Progress line per leaderboard row."""
    print(  # stdout-compat
        f"  {row.player:<18} reward={row.reward_ep!s:>9} "
        f"evals={row.evals_ep!s:>7} AP={row.ap!s:>7} "
        f"ratio={row.ratio!s:>7} {row.notes}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the boards, emit the leaderboard."""
    p = argparse.ArgumentParser(prog="bench.players.run")
    p.add_argument(
        "--players",
        default="",
        help="comma-separated player names (default: every "
        "registered player each board supports)",
    )
    p.add_argument(
        "--boards",
        default="meta,open,contraction",
        help=f"comma-separated boards (default meta,open,"
        f"contraction; known: {', '.join(ALL_BOARDS)})",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--meta-cases", type=int, default=META_CASES)
    p.add_argument("--meta-budget", type=int, default=META_BUDGET)
    p.add_argument("--open-pages", type=int, default=OPEN_PAGES)
    p.add_argument("--open-page", type=int, default=OPEN_PAGE)
    p.add_argument("--open-budget", type=int, default=OPEN_BUDGET)
    p.add_argument("--neural-epochs", type=int, default=NEURAL_EPOCHS)
    p.add_argument("--neural-hidden", type=int, default=NEURAL_HIDDEN)
    p.add_argument(
        "--scales",
        default=CONTRACTION_SCALES,
        help="comma-separated tensor counts for the contraction board",
    )
    p.add_argument(
        "--instances",
        type=int,
        default=CONTRACTION_INSTANCES,
        help="bond-network instances per contraction scale",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=CONTRACTION_SAMPLES,
        help="rollouts per board for sampled contraction players",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="shrink every knob — the seconds-scale lane",
    )
    p.add_argument(
        "--pin",
        action="store_true",
        help="pin the table as bench/baselines/leaderboard_players.json",
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
        "--out",
        type=Path,
        default=Path("docs"),
        help="leaderboard output directory (default docs/)",
    )
    p.add_argument(
        "--list-players",
        action="store_true",
        help="print the registered roster and exit",
    )
    args = p.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    if args.list_players:
        for name in registry.names():
            s = registry.spec(name)
            print(f"{name:<18} {s.kind:<9} {', '.join(s.boards)}")
        return 0

    if args.smoke:
        for key, val in SMOKE.items():
            if hasattr(args, key):
                setattr(args, key, val)

    players = [
        x.strip() for x in args.players.split(",") if x.strip()
    ] or None
    boards = [x.strip() for x in args.boards.split(",") if x.strip()]
    unknown = [b for b in boards if b not in ALL_BOARDS]
    if unknown:
        print(
            f"unknown board(s): {', '.join(unknown)} — "
            f"known: {', '.join(ALL_BOARDS)}"
        )
        return 2
    try:
        scales = tuple(
            int(x) for x in args.scales.split(",") if x.strip()
        )
    except ValueError:
        print(f"bad --scales {args.scales!r} — want e.g. '8,12,16'")
        return 2
    if not scales:
        print("bad --scales — empty")
        return 2

    protocol = {
        "seed": args.seed,
        "players": players or "all-registered",
        "meta_cases": args.meta_cases,
        "meta_budget": args.meta_budget,
        "meta_held_out": "gen_cases(seed+7919)",
        "open_pages": args.open_pages,
        "open_page": args.open_page,
        "open_budget": args.open_budget,
        "open_split": "odd pages eval / even pages train (neural)",
        "oracle_pairing": "memoized per-candidate-seeded verdicts",
        "neural_epochs": args.neural_epochs,
        "neural_hidden": args.neural_hidden,
        "contraction_scales": list(scales),
        "contraction_instances": args.instances,
        "contraction_samples": args.samples,
    }

    rows = []
    board_meta: dict = {}
    for board in boards:
        print(f"== board {board} ==", flush=True)
        try:
            board_rows, meta = run_board(
                board,
                players,
                seed=args.seed,
                meta_cases=args.meta_cases,
                meta_budget=args.meta_budget,
                open_pages=args.open_pages,
                open_page=args.open_page,
                open_budget=args.open_budget,
                neural_epochs=args.neural_epochs,
                neural_hidden=args.neural_hidden,
                scales=scales,
                instances=args.instances,
                samples=args.samples,
                device=args.device,
            )
        except ValueError as e:
            print(f"  skipped — {e}", flush=True)
            continue
        rows.extend(board_rows)
        board_meta[board] = meta
        for r in board_rows:
            _print_row(r)

    md_path = args.out / "leaderboard_players.md"
    json_path = args.out / "leaderboard_players.json"
    write_outputs(
        rows,
        md_path=md_path,
        json_path=json_path,
        protocol=protocol,
        meta=board_meta,
        device=args.device,
    )
    from bench.benchkit import table as _t

    record = _t.table_record(
        "leaderboard_players",
        (
            {
                "name": f"{r.board}/{r.player}",
                "metrics": {"reward_ep": r.reward_ep, "ap": r.ap},
            }
            for r in rows
        ),
        directions={"reward_ep": "down", "ap": "down"},
        findings=[
            {
                "claim": "registered players compared under one "
                "paired protocol",
                "verdict": "parity",
                "headline": f"{len(rows)} rows",
            }
        ],
        provenance={"quick": bool(args.smoke)},
    )
    if not args.no_ledger:
        _t.append_record(record)
    if args.pin:
        print(
            f"  -> pinned {_t.write_table_baseline(record, args.baselines)}"
        )
    print(f"leaderboard: {len(rows)} rows", flush=True)
    print(f"  -> {md_path}")
    print(f"  -> {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
