"""Perf-parity probe: ``handl-egraph`` vs Python ``EGraph`` on the zoo.

Plan 0033's hard gate — "``handl-egraph`` saturation time/memory
within 2x of ``EGraph`` on the 68-module corpus" — measured, not
asserted.  Each held-out ``mlopt_discovery.zoo`` workload is lifted
to an IR term (``TorchSource().to_ir``), then saturated twice:

* **Python** — ``mlopt_core.egraph.EGraph.run`` in-process, timed
  with ``perf_counter``; peak Python-heap bytes via ``tracemalloc``
  (a second, untraced run supplies the honest wall time — tracing
  perturbs it);
* **Rust** — the same spec through ``mlopt_discovery.handl_egraph``
  (``egraph-run`` subprocess).  Wall time is ``perf_counter``; peak
  RSS is the child's own ``VmHWM`` polled from ``/proc`` — per-module,
  post-exec, immune to the parent's resident set.

Both sides run the *identical* spec — the 84 declarative ``vocab/``
rewrite rows (``check``/``derive``-hooked rows stay Python by the
phase-5 boundary) under the audit fixtures' bounding convention
(``max_iterations=40``, ``max_nodes=20_000``, ``patience=3``, the
four comm/assoc laws budgeted at 300 applications each — the same
numbers ``egraph_differential.py`` uses so saturation stays a
comparison, not a blow-up contest).

Parity evidence rides along: the engines' stats fields
(``stop``/``iterations``/``n_enodes``/``n_classes``/
``n_proof_edges``/``truncation_level``) are compared per module — a
stats divergence is reported as a finding next to the timing row.

Usage::

    .venv/bin/python bench/drivers/audit/egraph_perf_parity.py
    .venv/bin/python bench/drivers/audit/egraph_perf_parity.py --limit 5
    .venv/bin/python bench/drivers/audit/egraph_perf_parity.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

#: The bounding convention — identical to egraph_differential's
#: compound fixtures so the corpus run measures the same contract.
OPTIONS = {
    "max_iterations": 40,
    "max_nodes": 20_000,
    "patience": 3,
    "rule_budgets": {
        "comm_add": 300,
        "comm_mul": 300,
        "assoc_add": 300,
        "assoc_mul": 300,
    },
}
COSTS = {"default": 1.0, "leaf": 0.0}

#: Per-module Rust timeout — a hung native run is a finding, not a
#: reason to wait forever.
RUST_TIMEOUT_S = 120.0

#: Stats fields compared between the two engines' reports.
_STATS_FIELDS = (
    "stop",
    "iterations",
    "n_enodes",
    "n_classes",
    "n_proof_edges",
    "truncation_level",
)


def _proc_vm_hwm_kb(pid: int) -> int | None:
    """Read ``VmHWM`` (peak RSS, kB) from ``/proc/<pid>/status``."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _rust_run_timed(
    bin_path: Path, spec: dict, timeout: float = RUST_TIMEOUT_S
) -> tuple[dict | None, float, int | None, str | None]:
    """Run ``egraph-run`` on *spec*; return (report, secs, rss_kb, err).

    All three stdio channels are temp files — no pipe can deadlock
    against a large spec or report.  Peak RSS comes from polling the
    child's ``VmHWM`` in ``/proc``: it is the child's own high-water
    mark *post-exec*, immune to the parent's resident set (a
    ``wait4``/``ru_maxrss`` reading would momentarily count the
    ~430 MB torch-loaded parent at spawn — measured, not guessed).
    A run shorter than the poll interval can be missed; ``rss_kb`` is
    then ``None`` and the row says so honestly.
    """
    # ``delete=True`` is safe: the files close only after ``proc.wait``
    # — the child is done reading/writing by then.
    with (
        tempfile.NamedTemporaryFile(
            "w+b", prefix="egraph-spec-", suffix=".json"
        ) as spec_f,
        tempfile.NamedTemporaryFile(
            "w+b", prefix="egraph-out-", suffix=".json"
        ) as out_f,
        tempfile.NamedTemporaryFile(
            "w+b", prefix="egraph-err-", suffix=".txt"
        ) as err_f,
    ):
        spec_f.write(json.dumps(spec).encode())
        spec_f.flush()
        # the child shares this open file description — rewind so it
        # reads from byte 0, not the post-write EOF.
        spec_f.seek(0)
        t0 = time.perf_counter()
        proc = subprocess.Popen(
            [str(bin_path)],
            stdin=spec_f,
            stdout=out_f,
            stderr=err_f,
        )
        peak: int | None = None
        deadline = time.monotonic() + timeout
        timed_out = False
        while proc.poll() is None:
            rss = _proc_vm_hwm_kb(proc.pid)
            if rss is not None:
                peak = rss if peak is None else max(peak, rss)
            if time.monotonic() > deadline:
                timed_out = True
                proc.kill()
                break
            time.sleep(0.002)
        proc.wait()
        secs = time.perf_counter() - t0
        out_f.seek(0)
        err_f.seek(0)
        stdout = out_f.read()
        stderr = err_f.read().decode(errors="replace")
    if timed_out:
        return None, secs, peak, "timeout"
    if proc.returncode != 0:
        return (
            None,
            secs,
            peak,
            f"exit={proc.returncode}: {stderr.strip()[:200]}",
        )
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as e:
        return None, secs, peak, f"bad json: {e}"
    return report, secs, peak, None


def _python_run(term, rules):
    """Saturate *term* under *rules*; return (stats, secs, peak_bytes).

    Two passes: the timed pass runs untraced (tracemalloc perturbs
    allocation-heavy code); the memory pass re-runs under tracemalloc
    for the peak.  EGraph runs are deterministic, so the second pass
    measures the same work.
    """
    from mlopt_core.egraph import EGraph

    g = EGraph()
    root = g.add_term(term)
    t0 = time.perf_counter()
    stats = g.run(
        list(rules),
        root,
        max_iterations=OPTIONS["max_iterations"],
        max_nodes=OPTIONS["max_nodes"],
        patience=OPTIONS["patience"],
        stop="fixed_point",
        rule_budgets=OPTIONS["rule_budgets"],
        cost_fn=lambda t: 0.0,
    )
    secs = time.perf_counter() - t0

    tracemalloc.start()
    g2 = EGraph()
    root2 = g2.add_term(term)
    g2.run(
        list(rules),
        root2,
        max_iterations=OPTIONS["max_iterations"],
        max_nodes=OPTIONS["max_nodes"],
        patience=OPTIONS["patience"],
        stop="fixed_point",
        rule_budgets=OPTIONS["rule_budgets"],
        cost_fn=lambda t: 0.0,
    )
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return stats, secs, peak


def _lift(workload) -> tuple[object | None, str | None]:
    """Lift a zoo workload to its IR root term."""
    import torch
    from mlopt_torch.adapters import TorchSource

    model, feed = workload.build()
    model = model.to(torch.float64)
    if isinstance(feed, torch.Tensor):
        feed = feed.to(torch.float64)
    elif isinstance(feed, (tuple, list)):
        feed = tuple(
            f.to(torch.float64) if isinstance(f, torch.Tensor) else f
            for f in feed
        )
    ir, _leaves = TorchSource().to_ir(model, feed)
    return ir.root, None


def main(argv: list[str] | None = None) -> int:
    """Run the corpus parity measurement; print the report table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--module", type=str, default=None)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--b", type=int, default=2)
    args = ap.parse_args(argv)

    from mlopt_discovery import handl_egraph as he
    from mlopt_discovery import zoo

    bin_path = he.egraph_run_bin()
    if bin_path is None:
        print(
            "egraph-run not available — build it:\n"
            "  cargo build -p mlopt-kit --bin egraph-run",
            file=sys.stderr,
        )
        return 2

    vocab = he.vocab_rewrites()
    rules = list(vocab.rules)
    print(
        f"vocab: {len(rules)} declarative rules "
        f"({vocab.n_skipped} hooked rows stay Python, "
        f"{vocab.epsilon} epsilon)"
    )

    workloads = zoo.zoo(d=args.d, t=args.t, b=args.b)
    if args.module:
        workloads = [w for w in workloads if w.name == args.module]
    if args.limit is not None:
        workloads = workloads[: args.limit]

    rows = []
    lift_fail = []
    stat_divs = []
    for wl in workloads:
        try:
            term, err = _lift(wl)
        except Exception as e:  # lift failure is a finding too
            lift_fail.append((wl.name, f"{type(e).__name__}: {e}"))
            print(f"  LIFT-FAIL {wl.name}: {e}")
            continue
        if term is None:
            lift_fail.append((wl.name, err or "no term"))
            continue

        # Rust side first (bounded), then Python.
        spec = he.build_spec(
            f"zoo:{wl.name}",
            [term],
            rules,
            options=OPTIONS,
            costs=COSTS,
        )
        try:
            rs_rep, rs_secs, rs_rss, rs_err = _rust_run_timed(
                Path(bin_path), spec
            )
        except Exception as e:
            rs_rep, rs_secs, rs_rss, rs_err = (
                None,
                None,
                None,
                f"{type(e).__name__}: {e}",
            )
        try:
            py_stats, py_secs, py_peak = _python_run(term, rules)
        except Exception as e:
            py_stats, py_secs, py_peak = (
                None,
                None,
                None,
            )
            print(f"  PY-FAIL {wl.name}: {e}")
            continue

        row = {
            "module": wl.name,
            "py_secs": py_secs,
            "rs_secs": rs_secs,
            "ratio": (rs_secs / py_secs)
            if (rs_secs and py_secs)
            else None,
            "py_peak_b": py_peak,
            "rs_rss_kb": rs_rss,
            "py_stats": {k: py_stats.get(k) for k in _STATS_FIELDS}
            if py_stats
            else None,
            "rs_stats": {k: rs_rep.get(k) for k in _STATS_FIELDS}
            if rs_rep
            else None,
            "rs_error": rs_err,
        }
        rows.append(row)
        div = (
            row["py_stats"] != row["rs_stats"]
            if rs_rep
            else "no-report"
        )
        if div is True:
            stat_divs.append(wl.name)
        flag = (
            ""
            if div is False
            else f"  stats={'DIV' if div is True else div}"
        )
        if rs_err:
            flag += f"  rs_err={rs_err}"
        print(
            f"  {wl.name:28s} py={py_secs:8.3f}s rs={rs_secs if rs_secs is not None else -1:8.3f}s"
            f" ratio={row['ratio'] if row['ratio'] is not None else -1:6.2f}"
            f" pyPk={py_peak / 1e6 if py_peak else 0:7.1f}MB"
            f" rsRSS={rs_rss if rs_rss else 0:8d}KB{flag}"
        )

    ok = [r for r in rows if r["ratio"] is not None]
    py_total = sum(r["py_secs"] for r in ok)
    rs_total = sum(r["rs_secs"] for r in ok)
    worst = max(ok, key=lambda r: r["ratio"], default=None)
    print()
    print("egraph perf parity — vocab bank over the zoo corpus")
    print(f"  modules measured : {len(ok)}")
    print(f"  lift failures    : {len(lift_fail)}")
    for n, e in lift_fail[:10]:
        print(f"    - {n}: {e}")
    print(f"  stats divergences: {len(stat_divs)}")
    for n in stat_divs[:10]:
        print(f"    - {n}")
    print(f"  python total     : {py_total:.3f}s")
    print(f"  rust total       : {rs_total:.3f}s")
    print(
        f"  aggregate ratio  : {rs_total / py_total:.2f}x"
        if py_total
        else "  aggregate ratio  : n/a"
    )
    if worst:
        print(
            f"  worst module     : {worst['module']} "
            f"({worst['ratio']:.2f}x)"
        )
    print(
        f"  python peak heap : {max((r['py_peak_b'] or 0) for r in rows) / 1e6:.1f}MB"
        if rows
        else ""
    )
    print(
        f"  rust peak RSS    : {max((r['rs_rss_kb'] or 0) for r in rows)}KB"
        if rows
        else ""
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "options": OPTIONS,
                    "costs": COSTS,
                    "n_rules": len(rules),
                    "n_skipped": vocab.n_skipped,
                    "rows": rows,
                    "lift_failures": lift_fail,
                    "stat_divergences": stat_divs,
                },
                indent=2,
            )
        )
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
