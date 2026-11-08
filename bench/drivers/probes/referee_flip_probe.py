"""Referee flip-rate probe — ``profitable`` keep-set stability.

The T-009/T-010 follow-up: ``profitable``'s documented margin noise
is offer-visible — one contaminated sweep dropped chainfuse's whole
claim set (1.37x vs compile delivered) while a fresh process kept
``gen_0,1,3`` and the 2.75x win.  This driver measures the flip rate
directly: each invocation is ONE fresh process, it runs the same
gate the board runs, and prints the kept-claim set as JSON.  Run it
N times and diff the outputs.

Modes:

* ``--gen`` — build the domain's eval cases (``_gen_cases`` at the
  eval seed) and run ``play._claim_cost_gate`` on each — the exact
  call ``_gen_board`` makes before the board's handler store is
  built.  ``--only`` filters case names (substring match).
* ``--zoo NAME[,NAME...]`` — build ``play._zoo_case`` for the named
  held-out workloads; the intake ``profitable`` sweep over every
  minted tier is what filters ``ex["gen_handlers"]``.

With ``--log-ratios`` the per-handler referee timings
(``genkernel.PROFITABLE_LOG``) are printed too — one row per timing
sample ``(name, t_spelled, t_kernel, window, keep_vote)``; a
borderline handler contributes up to three consecutive rows and its
decision is their majority — so borderline specs can be correlated
with the keep/drop outcome.

Usage (one sample per process)::

    .venv/bin/python bench/drivers/probes/referee_flip_probe.py \
        --gen --only chainfuse,twosite,chain_mlp
    .venv/bin/python bench/drivers/probes/referee_flip_probe.py \
        --zoo ChunkedSSD
"""

from __future__ import annotations

import argparse
import json
import sys


def _gen_sample(seed: int, n: int, only: list[str]) -> dict:
    """Run the offer cost gate over the gen eval cases; return keep sets."""
    from mlopt_discovery import play as P

    cases = P._gen_cases(seed, n)
    out: dict[str, dict] = {}
    for case in cases:
        name = case[3]
        if only and not any(o in name for o in only):
            continue
        ex = case[5]
        minted = sorted(ex["gen_handlers"])
        P._claim_cost_gate(case)
        out[case[0]] = {
            "name": name,
            "minted": minted,
            "kept": sorted(ex["gen_handlers"]),
        }
    return out


def _zoo_sample(names: list[str], d: int, t: int, b: int) -> dict:
    """Build zoo cases; return the post-referee gen-handler sets."""
    from mlopt_discovery import play as P
    from mlopt_discovery import zoo

    wls = {w.name: w for w in zoo.zoo(d, t, b)}
    out: dict[str, dict] = {}
    for nm in names:
        wl = wls.get(nm)
        if wl is None:
            out[nm] = {"error": "no such workload"}
            continue
        try:
            case = P._zoo_case(wl)
        except Exception as exc:  # honest decline, same as zoo_probe
            out[nm] = {
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        out[nm] = {
            "kept": sorted(case[5]["gen_handlers"]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", action="store_true", help="gen claim gate")
    ap.add_argument("--zoo", type=str, default=None, help="zoo names")
    ap.add_argument(
        "--only", type=str, default="", help="name substrings"
    )
    ap.add_argument("--seed", type=int, default=7919)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--zoo-d", type=int, default=16)
    ap.add_argument("--zoo-t", type=int, default=8)
    ap.add_argument("--zoo-b", type=int, default=2)
    ap.add_argument("--log-ratios", action="store_true")
    args = ap.parse_args(argv)

    if args.gen:
        sample = _gen_sample(
            args.seed, args.n, [s for s in args.only.split(",") if s]
        )
    elif args.zoo:
        sample = _zoo_sample(
            [s for s in args.zoo.split(",") if s],
            args.zoo_d,
            args.zoo_t,
            args.zoo_b,
        )
    else:
        ap.error("pass --gen or --zoo")
        return 2

    result = {"sample": sample}
    if args.log_ratios:
        from mlopt_discovery import genkernel as gk

        result["ratios"] = list(getattr(gk, "PROFITABLE_LOG", ()))
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
