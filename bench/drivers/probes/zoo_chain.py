"""Chain-level pay on the **real** zoo corpus: does `gen` deliver on modules?

`--probe` measures two fixed cases (`chainfuse`, `mulsoftplus`) and gets
2.87x.  That is a whole-chain kernel launched once over a synthetic tape.
The question this driver answers is whether the same machinery pays on the
held-out model zoo, at wide shapes, on real module boundaries.

Each workload goes through the shipped path unchanged: build -> intake
dtype -> `TorchSource.to_ir` -> mint tiers -> `genkernel.profitable` (the
measured cost referee) -> `gen_sink` -> `play.gen_probe` (deliver, verify,
time vs eager and `torch.compile`).  No special casing, so the row is the
number a caller would get.

**Dtype is a variable, and it moves the answer.**  ``--dtype fp64`` (the
default, the intake convention every earlier table used) and ``--dtype fp32``
(what a caller ships) disagree on 12 of 65 modules: 23 beat eager in every
round in fp32, 17 in fp64.  The same IR folds the same way in both and a fold's
sign is set by the backend kernel at that width — `RMSNormPerHead` is 1.5x in
fp32 and 0.21x in fp64 on identical delivered structure
(`bench/drivers/probes/zoo_dtype.py`).

Wide shapes matter: the zoo's default is ``d16 t8 b2``, where the kernels
are too small for the launcher to matter.  ``--d 64 --t 32 --b 8`` is the
shape the module probe uses.

**Sample, not a sweep.**  `torch.compile` is the baseline, so each workload
costs tens of seconds at wide shapes and a handful time out on this box.
Select workloads with ``--only``; a partial table is reported as partial.
``--budget`` (default 150 s) caps each workload: one module whose compile never
returns used to stop the run mid-table and hide every workload after it, so
over-budget rows are now recorded as rows.  ``--only ''`` measures all 68.

**Rounds, not one shot.**  Each row is the median of ``--rounds`` (default 3)
independent rounds, with the spread kept.  Timing an op-level ratio to two
decimals is fine; timing two *modules* at different moments is not — an
unrelated process on the box moved one workload's ratio from 0.42x to 1.44x
without touching a line of code.  Run this on an idle machine.

Run::

    .venv/bin/python -m bench.drivers.probes.zoo_chain \\
        --only GatedAttentionUnit,MLPMixerBlock

    # the sweep a caller would see (fp32), isolated, one round per row
    .venv/bin/python -m bench.drivers.probes.zoo_chain \\
        --only '' --isolate --dtype fp32 \\
        --out docs/reports/zoo_chain_sweep_fp32.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

from mlopt_discovery import zoo as zoo_mod
from mlopt_discovery.play import _zoo_case, gen_probe

DEFAULT_ONLY = "GatedAttentionUnit,MLPMixerBlock,BlockSparseAttention,TimeCondConv,FNOBlock"
DEFAULT_BUDGET = 150.0


class _OverBudget(Exception):
    """A single workload exceeded its wall-clock budget."""


@contextmanager
def _budget(seconds: float):
    """Abort the guarded block after ``seconds`` (0 or less = no limit)."""
    if seconds <= 0:
        yield
        return

    def _boom(_signum, _frame):
        raise _OverBudget(f"exceeded {seconds:g}s")

    prev = signal.signal(signal.SIGALRM, _boom)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def measure(
    *,
    d: int,
    t: int,
    b: int,
    only: tuple[str, ...],
    reps: int,
    rounds: int = 3,
    budget: float = DEFAULT_BUDGET,
    dtype: str = "fp64",
) -> list[dict]:
    """Run the shipped gen path over the named zoo workloads.

    Each workload is measured ``rounds`` times and reported by the
    **median** ratio plus its spread.  One round is not evidence: the
    two arms are timed at different moments, so a clock or load
    excursion during either arm moves the ratio by tens of percent with
    no code change (measured: one workload read 0.42x and 1.44x across
    runs that differed only in what else the box was running).

    Each workload also carries a wall-clock ``budget`` (default 150 s,
    ``0`` disables it).  A module whose ``torch.compile`` never returns
    would otherwise stop the sweep at that row and report nothing for
    every workload after it — the earlier run wedged for minutes on one
    module and produced no table at all.  Over-budget rows are recorded
    as such, so the table stays complete and says which rows are
    missing rather than silently ending early.
    """
    want = set(only)
    rows: list[dict] = []
    for wl in zoo_mod.zoo(d, t, b):
        if want and wl.name not in want:
            continue
        try:
            with _budget(budget):
                row = _measure_one(
                    wl, reps=reps, rounds=rounds, dtype=dtype
                )
        except _OverBudget:
            rows.append(
                {
                    "case": f"zoo:{wl.name}",
                    "delivered": False,
                    "note": f"over budget: no result within {budget:g}s "
                    "(`torch.compile` on this module did not return)",
                }
            )
            print(
                f"zoo[{wl.name}]: over budget ({budget:g}s)", flush=True
            )
            continue
        except Exception as exc:
            rows.append(
                {
                    "case": f"zoo:{wl.name}",
                    "delivered": False,
                    "note": f"export declined: {type(exc).__name__}: {exc}",
                }
            )
            print(f"zoo[{wl.name}]: export declined: {exc}", flush=True)
            continue
        if not row.get("delivered"):
            rows.append(row)
            print(f"zoo[{wl.name}]: undeliverable", flush=True)
            continue
        rows.append(row)
        print(f"zoo[{row['case']}]: {row}", flush=True)  # stdout-compat
    return rows


def _measure_one(
    wl, *, reps: int, rounds: int, dtype: str = "fp64"
) -> dict:
    """Time one workload ``rounds`` times; the row is the median ratio."""
    case = _zoo_case(wl, dtype=dtype)
    got = [gen_probe(case, reps=reps) for _ in range(rounds)]
    ok = [r for r in got if r.get("delivered")]
    if not ok:
        return dict(got[0], dtype=dtype)
    speeds = sorted(round(float(r["speedup"]), 3) for r in ok)
    row = dict(ok[0])
    # Every ratio keeps the pair it was computed from: rounds are
    # timed at different moments, so a ratio and a µs figure taken
    # from different rounds is not a measurement, it is an average
    # of two mistakes.
    row["runs"] = [
        {
            "speedup": round(float(r["speedup"]), 3),
            "eager_us": round(r["baseline_us"], 1),
            "compiled_us": round(r["compiled_us"], 1),
            "gen_us": round(r["gen_us"], 1),
        }
        for r in ok
    ]
    row["speedup"] = round(statistics.median(speeds), 3)
    row["spread"] = [speeds[0], speeds[-1]]
    for stale in (
        "baseline_us",
        "compiled_us",
        "gen_us",
        "vs_compiled",
    ):
        row.pop(stale, None)
    row["dtype"] = dtype
    row["verified"] = all(r.get("verified") for r in ok)
    row["max_abs"] = max(r.get("max_abs") or 0 for r in ok)
    return row


def _measure_isolated(
    names: list[str],
    *,
    d: int,
    t: int,
    b: int,
    reps: int,
    rounds: int,
    budget: float,
    dtype: str = "fp64",
) -> list[dict]:
    """Measure each workload in its own process, killing it if it overruns.

    A wall-clock alarm *inside* the parent cannot do this job: `torch.compile`
    handles exceptions internally, so a `SIGALRM` raised during compilation is
    swallowed by dynamo and then trips its own bookkeeping
    (`AttributeError: module 'torch._dynamo' has no attribute 'exc'`) — after
    which every later workload in the same process fails the same way.  A
    separate process per workload makes the timeout a `killpg`, so a module
    whose compile never returns costs one row instead of the whole table (and
    each row starts from a fresh CUDA context, which is also stricter).
    """
    rows: list[dict] = []
    for name in names:
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as fh:
            out = fh.name
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "bench.drivers.probes.zoo_chain",
            "--d",
            str(d),
            "--t",
            str(t),
            "--b",
            str(b),
            "--reps",
            str(reps),
            "--rounds",
            str(rounds),
            "--dtype",
            dtype,
            "--only",
            name,
            "--budget",
            "0",
            "--out",
            out,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=budget)
            payload = Path(out).read_text().strip()
            row = json.loads(payload)[0] if payload else None
            if row is None:
                row = {
                    "case": f"zoo:{name}",
                    "delivered": False,
                    "note": "child exited without a row",
                }
        except subprocess.TimeoutExpired:
            # inductor forks compile workers; take the whole session down.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            row = {
                "case": f"zoo:{name}",
                "delivered": False,
                "dtype": dtype,
                "note": f"over budget: killed after {budget:g}s",
            }
            print(f"zoo[{name}]: over budget ({budget:g}s)", flush=True)
        finally:
            Path(out).unlink(missing_ok=True)
        rows.append(row)
        if row.get("delivered"):
            print(f"zoo[{row['case']}]: {row}", flush=True)
        elif "over budget" not in str(row.get("note", "")):
            print(f"zoo[{name}]: {row.get('note')}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Parse the shape / selection flags and print the rows."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--t", type=int, default=32)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="timing rounds per workload; the row reports the median ratio",
    )
    ap.add_argument(
        "--only",
        default=DEFAULT_ONLY,
        help="comma-separated workload names; empty string = every workload",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET,
        help="wall-clock seconds per workload before the row is recorded as over budget (0 = no limit)",
    )
    ap.add_argument("--out", default="", help="optional JSON path")
    ap.add_argument(
        "--dtype",
        default="fp64",
        choices=("fp64", "fp32"),
        help="intake dtype: fp64 is the convention, fp32 is what a caller "
        "ships (they disagree on 12 of 65 modules)",
    )
    ap.add_argument(
        "--isolate",
        action="store_true",
        help="measure each workload in its own process so a hung compile costs one row, not the table",
    )
    args = ap.parse_args(argv)

    only = tuple(x for x in args.only.split(",") if x)
    if args.isolate:
        names = (
            list(only)
            if only
            else [wl.name for wl in zoo_mod.zoo(args.d, args.t, args.b)]
        )
        rows = _measure_isolated(
            names,
            d=args.d,
            t=args.t,
            b=args.b,
            reps=args.reps,
            rounds=args.rounds,
            budget=args.budget,
            dtype=args.dtype,
        )
    else:
        rows = measure(
            d=args.d,
            t=args.t,
            b=args.b,
            only=only,
            reps=args.reps,
            rounds=args.rounds,
            budget=args.budget,
            dtype=args.dtype,
        )
    won = [
        r
        for r in rows
        if r.get("delivered") and (r.get("speedup") or 0) > 1.0
    ]
    over = sum(
        1 for r in rows if "over budget" in str(r.get("note", ""))
    )
    print(
        f"\n{len(rows)} workloads: {len(won)} faster than eager, "
        f"{sum(1 for r in rows if not r.get('delivered'))} undeliverable"
        + (f" ({over} over budget)" if over else "")
    )
    if args.out:
        Path(args.out).write_text(
            json.dumps(rows, indent=1, default=str) + "\n"
        )
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
