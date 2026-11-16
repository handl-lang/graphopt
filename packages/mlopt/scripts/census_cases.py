"""Case census — build ``DeliveryCase``s cheaply and dump board facts.

The delivery-prior autopsy (host ``project/reports/
delivery_prior_autopsy.md`` §4) calls for a pre-screen pass before
paying for oracle drains: ``eager_med_ms`` / ``n_enodes`` /
``n_params`` / ``fusible`` / eligible-config count predict whether any
runner config can clear ``WIN_MARGIN``.  This driver is the
``census_cases_v5.py`` pattern generalized to a manifest:

    .venv/bin/python scripts/census_cases.py \
        --manifest /home/luis/tmp/delivery_v6/cases.json \
        --out /home/luis/tmp/delivery_v6/census.jsonl \
        --bench-iters 10

Manifest is a JSON list of ``{"name", "ref", "opts"}`` objects —
``opts`` are adapter ``load`` options merged over ``--adapter-opt``
globals (per-case wins), same convention as ``train_delivery_prior``'s
``REF@k=v`` suffix but JSON-native so ``leaves={...}`` shapes don't
fight shell quoting.  Each row is appended to ``--out`` as soon as the
case finishes, so an interrupted run keeps its measurements; a final
summary table prints at the end.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from train_delivery_prior import _adapter_opts, build_case

from mlopt.deliver_train import board_of

__all__ = ["main"]


def _row(
    case_spec: dict[str, Any],
    *,
    bench_iters: int,
    iters: int,
    verify: bool,
    device: str | None,
    drain: bool,
) -> dict[str, Any]:
    """Build one case and return its census row (never raises)."""
    name = case_spec.get("name") or case_spec["ref"]
    ref = case_spec["ref"]
    opts = dict(case_spec.get("opts") or {})
    t0 = time.time()
    row: dict[str, Any] = {"name": name, "ref": ref, "opts": opts}
    try:
        case = build_case(
            ref,
            name=name,
            opts=opts,
            bench_iters=bench_iters,
            verify=verify,
            max_iterations=iters,
        )
        if device:
            # Adapters without a ``device`` load opt (``hnl``/``arch``)
            # bind CPU tensors; the torch eval path moves params to the
            # input device per call, so moving the probe inputs is the
            # whole placement — the board then sees CUDA inputs and
            # offers the graph configs honestly.
            import torch

            def _mv(t: Any) -> Any:
                return t.to(device) if isinstance(t, torch.Tensor) else t

            ins = case.inputs
            case.inputs = tuple(_mv(t) for t in ins) if isinstance(ins, (tuple, list)) else _mv(ins)
            row["device_override"] = device
        board = board_of(case)
        st = board.state
        row.update(
            ok=True,
            elapsed_s=round(time.time() - t0, 1),
            pending=list(st.pending),
            n_eligible=len(st.pending),
            cuda=st.cuda,
            fusible=st.fusible,
            compile_ok=st.compile_ok,
            alias_spec=st.alias_spec,
            eager_min_ms=st.eager_min_ms,
            eager_med_ms=st.eager_med_ms,
            n_inputs=st.n_inputs,
            n_enodes=st.n_enodes,
            n_params=st.n_params,
            lowprec=st.lowprec,
        )
        if drain:
            # The expensive half: one scratch-board pass evaluates every
            # eligible config through the real referee (re-lower +
            # verify + paired bench).  Returns the oracle pick AND the
            # persisted per-config measurement records — the v6 answer
            # key the autopsy asked for.  A drain failure must not
            # sink the census row — record it and move on.
            from mlopt.deliver_train import oracle_drain

            try:
                pick, records = oracle_drain(case)
                row["oracle_pick"] = pick
                row["drain"] = records
            except Exception as e:  # noqa: BLE001
                row["drain_error"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001 — the census reports, not raises
        row.update(
            ok=False,
            elapsed_s=round(time.time() - t0, 1),
            error=f"{type(e).__name__}: {e}",
        )
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="JSON list of {name, ref, opts}")
    ap.add_argument("--out", required=True, help="JSONL output path (appended)")
    ap.add_argument("--adapter-opt", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--bench-iters", type=int, default=10)
    ap.add_argument("--iters", type=int, default=4, help="search max_iterations")
    ap.add_argument(
        "--device",
        default=None,
        help="post-build input placement override (e.g. cuda) for adapters "
        "without a device load opt — no-op when inputs are already there",
    )
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument(
        "--drain",
        action="store_true",
        help="after the census facts, run oracle_drain (lower+verify+"
        "paired bench per eligible config) and record pick + records",
    )
    args = ap.parse_args(argv)

    global_opts = _adapter_opts(args.adapter_opt)
    specs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if isinstance(specs, dict):
        specs = specs["cases"]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for spec in specs:
            merged = {**global_opts, **(spec.get("opts") or {})}
            spec = {**spec, "opts": merged}
            print(f"== census {spec.get('name') or spec['ref']} ({spec['ref']}) ==", flush=True)
            row = _row(
                spec,
                bench_iters=args.bench_iters,
                iters=args.iters,
                verify=not args.no_verify,
                device=args.device,
                drain=args.drain,
            )
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if row["ok"]:
                n_ok += 1
                print(
                    f"   OK {row['elapsed_s']}s pending={row['n_eligible']} "
                    f"fusible={row['fusible']} compile={row['compile_ok']} "
                    f"alias={row['alias_spec']} eager_med={row['eager_med_ms']} "
                    f"enodes={row['n_enodes']} params={row['n_params']} "
                    f"pick={row.get('oracle_pick', '—')}",
                    flush=True,
                )
            else:
                print(f"   FAIL {row['elapsed_s']}s {row['error']}", flush=True)
    print(f"== census done: {n_ok}/{len(specs)} cases built -> {out_path} ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
