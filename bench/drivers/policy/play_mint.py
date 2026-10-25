"""Mint during play — compose new rules from the board's own cases.

The G* loop needs the middle step the bank-growth experiment skipped:
rules that are minted *while playing*, admitted, and then handed to the
same players.  This driver does that middle step on the `search` domain's
own corpus:

1. **mint** — compose ordered pairs of the shipped bank over the board's
   cases (`mlopt_core.meta.synthesize_rules`, seed terms = the cases, one
   round, capped);
2. **admit** — three gates, in the shipped order: **pays**
   (`count_cost(rhs) < count_cost(lhs)`), **novel** (the bank cannot
   derive `lhs == rhs` — `verifier.verify_law`), and **fires** (applying
   it to a case term actually lowers extraction cost);
3. **hand over** — the admitted laws are written as `law_to_data` rows so
   `bench.drivers.policy.bank_growth --extra` can grow the bank with them
   and measure the frozen player.

Run::

    .venv/bin/python -m bench.drivers.policy.play_mint \
        --json /tmp/play_minted.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlopt_core.cost import count_cost
from mlopt_core.egraph import EGraph
from mlopt_core.laws import DEFAULT
from mlopt_core.laws.serialize import law_to_data
from mlopt_core.meta import synthesize_rules
from mlopt_discovery import play as P
from mlopt_discovery.verifier import verify_law


def mint(
    *,
    seed: int = 0,
    cases: int = 32,
    fuel: int = 220,
    cap: int = 60,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Mint candidates from the board's cases; return admitted + rejects."""
    domain = P.DOMAINS["search"]
    train = domain.cases(seed, cases)
    terms = [case[1] for case in train]
    cands = synthesize_rules(
        DEFAULT,
        seed_terms=terms,
        fuel=fuel,
        numeric_check=True,
        rounds=1,
        max_emitted=cap,
    )
    admitted: list[Any] = []
    rows: list[dict[str, Any]] = []
    for rule in cands:
        row: dict[str, Any] = {"name": rule.name}
        try:
            pays = count_cost(rule.rhs) < count_cost(rule.lhs)
        except Exception as exc:  # unpriceable candidate
            row.update(admitted=False, why=f"cost: {exc}")
            rows.append(row)
            continue
        if not pays:
            row.update(admitted=False, why="no pay")
            rows.append(row)
            continue
        try:
            derivable = verify_law(
                rule.lhs, rule.rhs, DEFAULT
            ).derivable
        except Exception as exc:  # verifier refused to run
            row.update(admitted=False, why=f"verify: {exc}")
            rows.append(row)
            continue
        if derivable:
            row.update(admitted=False, why="derivable from the bank")
            rows.append(row)
            continue
        fires = _fires(rule, terms)
        if not fires:
            row.update(admitted=False, why="never fires on a case")
            rows.append(row)
            continue
        row.update(admitted=True, why="cleared", fires=fires)
        rows.append(row)
        admitted.append(rule)
    return admitted, rows


def _fires(rule: Any, terms: list[Any]) -> int:
    """How many case terms the rule fires on and lowers in one pass."""
    hits = 0
    for term in terms:
        eg = EGraph()
        root = eg.add_term(term)
        try:
            eg.rebuild()
            before = eg.extract_best(root, count_cost)
        except Exception:
            continue
        stats = eg.run([rule], root, max_iterations=1)
        if not stats.get("fired"):
            continue
        try:
            eg.rebuild()
            if eg.extract_best(root, count_cost) < before:
                hits += 1
        except Exception:
            continue
    return hits


def main(argv: list[str] | None = None) -> int:
    """Mint from the board's cases, admit, and print/write the result."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cases", type=int, default=32)
    ap.add_argument("--fuel", type=int, default=220)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument(
        "--json", help="write admitted laws as law_to_data rows"
    )
    ap.add_argument("--report", help="write the full admit table here")
    args = ap.parse_args(argv)

    admitted, rows = mint(
        seed=args.seed, cases=args.cases, fuel=args.fuel, cap=args.cap
    )
    cleared = [r for r in rows if r.get("admitted")]
    print(
        f"minted {len(rows)} · admitted {len(cleared)} · "
        f"rejects {len(rows) - len(cleared)}"
    )
    for row in rows:
        mark = "admit" if row.get("admitted") else "reject"
        extra = f" fires={row['fires']}" if row.get("fires") else ""
        print(
            f"  {mark:6} {row['name']:30} {row.get('why', '')}{extra}"
        )
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([law_to_data(r) for r in admitted], indent=1),
            encoding="utf-8",
        )
        print(f"\nwrote {len(admitted)} admitted laws -> {path}")
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"wrote the admit table -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
