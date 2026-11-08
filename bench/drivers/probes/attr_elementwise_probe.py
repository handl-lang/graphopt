"""Attr-bearing elementwise bodies — measure the real spec path.

The T-010 census found **zero** attr-carrying elementwise ops across
all 68 zoo workloads *within* ``genkernel.ELEMENTWISE`` — but a
re-census shows the corpus DOES carry attr-bearing elementwise ops:
``clamp``/``clamp_min``/``clamp_max`` sites with concrete ``min``/
``max`` attrs in 11 workloads (CosineAttention, GCNLayer,
AffineCoupling, CapsuleRouting, ChunkedRetention, RWKVMixBlock,
STEQuantLinear, RelPosTableAttention, SinkWindowAttention,
TaylorLinearAttention, TernaryWeightLinear).  They were excluded
only because the alphabet was a deliberate narrow claim — the
widening this probe first measured in-process has since shipped.

This driver measures what the *fixed* attr machinery does with them.
It was first run with ``genkernel.ELEMENTWISE`` **widened in-process**
by the three clamp ops; the widening has since shipped — the probe
now runs the unmodified intake path: ``play._zoo_case`` mints
``gen_*`` specs whose canonical form carries attr dicts,
``profitable`` prices them through spelled eval
(``_eval_spec`` → ``_IR_TO_TORCH``'s attr-aware clamp bindings), the
cost gate drops losers, and ``play.gen_probe`` delivers kept claims
through ``hybrid_sink`` — where ``_triton_node_h`` declines the attr
dict and the ``torch.compile``d spelled body ships, exactly the
posture the retro recorded.

Run::

    .venv/bin/python -m bench.drivers.probes.attr_elementwise_probe \\
        --dtype fp64 --probe --out docs/reports/attr_elementwise_probe.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from mlopt_discovery import genkernel as gk
from mlopt_discovery import play
from mlopt_discovery import zoo as zoo_mod

#: The attr-bearing elementwise ops the zoo corpus actually spells.
#: Shipped in ``genkernel.ELEMENTWISE`` (the in-process widening this
#: probe first measured has landed); kept here as the census record.
CLAMP_OPS = frozenset({"clamp", "clamp_min", "clamp_max"})


def _has_attr_dict(spec: Any) -> bool:
    """True when a canonical spec carries an attr dict anywhere."""
    if isinstance(spec, (tuple, list)):
        return any(
            isinstance(e, dict) or _has_attr_dict(e) for e in spec[1:]
        )
    return False


def _attr_sites(spec: Any) -> list:
    """Every ``(op, attrs)`` pair a spec carries, recursively.

    Within a ``gen_*`` spec every dict IS an elementwise-body attr by
    construction — the fusion pattern abstracts non-elementwise
    children to metavars, so nothing else can hold one.  (The
    ``scan_*``/``red_*`` tiers' head attrs — ``applyd``'s
    ``kind``/``axis``, a reduction's ``dim`` — are a different,
    already-measured slot and are excluded by the ``gen_`` tag
    filter in :func:`audit`.)
    """
    out: list = []
    if not isinstance(spec, (tuple, list)):
        return out
    attrs = [e for e in spec[1:] if isinstance(e, dict)]
    if attrs:
        out.append((spec[0], attrs))
    for e in spec[1:]:
        if not isinstance(e, dict):
            out.extend(_attr_sites(e))
    return out


def _spec_summary(spec: Any) -> dict:
    """Compact description of a minted spec for the report."""
    return {
        "head": spec[0] if isinstance(spec, (tuple, list)) else spec,
        "attr_sites": _attr_sites(spec),
        "pattern": repr(spec),
    }


def audit(wl: Any, dtype: str) -> dict:
    """Run one (workload, dtype) cell under the widened alphabet."""
    n0 = len(play.TRITON_REFEREE_LOG)
    n1 = len(gk.PROFITABLE_LOG)
    # ``_zoo_case`` gates ``gen_handlers`` through ``profitable`` —
    # the post-intake table holds only kept claims.  Spy on
    # ``_mint_tiers`` to capture the pre-gate mint so dropped attr
    # specs are still reported.
    minted: dict = {}
    orig_mint = play._mint_tiers

    def spy(root: Any, covered: set) -> dict:
        h = orig_mint(root, covered)
        minted.update(h)
        return h

    play._mint_tiers = spy
    try:
        case = play._zoo_case(wl, dtype=dtype)
    finally:
        play._mint_tiers = orig_mint
    genh = case[5]["gen_handlers"]
    attr_specs = {
        tag: _spec_summary(h["pattern"])
        for tag, h in minted.items()
        if tag.startswith("gen_") and _has_attr_dict(h.get("pattern"))
    }
    kept = sorted(genh)
    samples: dict[str, list] = {}
    for name, ts, tk, window, vote in gk.PROFITABLE_LOG[n1:]:
        samples.setdefault(name, []).append(
            {
                "window": window,
                "t_spelled_us": round(ts * 1e6, 3),
                "t_kernel_us": round(tk * 1e6, 3),
                "ratio_kernel_over_spelled": round(tk / ts, 4)
                if ts > 0
                else None,
                "keep_vote": bool(vote),
            }
        )
    consults = [
        {
            "tag": r.get("tag"),
            "kernel": r.get("kernel"),
            "pattern_head": r.get("pattern_head"),
            "outcome": r.get("outcome"),
            "attempts": r.get("attempts"),
            "reason": r.get("reason"),
            "samples": samples.get(r.get("tag"), []),
            "kept_by_referee": r.get("tag") in kept,
        }
        for r in play.TRITON_REFEREE_LOG[n0:]
    ]
    return {
        "workload": wl.name,
        "dtype": dtype,
        "gen_specs_minted": len(minted),
        "attr_specs": attr_specs,
        "attr_specs_kept": sorted(t for t in attr_specs if t in genh),
        "kept": kept,
        "consults": consults,
        "case": case,
    }


def measure(
    *,
    dtype: str,
    probe: bool,
    d: int = 16,
    t: int = 8,
    b: int = 2,
    workloads: set[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for wl in zoo_mod.zoo(d, t, b):
        if workloads is not None and wl.name not in workloads:
            continue
        try:
            row = audit(wl, dtype)
        except Exception as exc:
            row = {
                "workload": wl.name,
                "dtype": dtype,
                "error": f"{type(exc).__name__}: {exc}",
            }
        case = row.pop("case", None)
        if probe and row.get("attr_specs_kept") and case is not None:
            try:
                row["gen_probe"] = play.gen_probe(case, reps=100)
            except Exception as exc:
                row["gen_probe"] = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
        rows.append(row)
        minted = ", ".join(
            f"{t}({s['head']},attr_sites={s['attr_sites']})"
            for t, s in row.get("attr_specs", {}).items()
        )
        kept_attr = row.get("attr_specs_kept") or []
        print(
            f"zoo:{wl.name}[{dtype}]: attr_specs={minted or 'none'} "
            f"attr_kept={kept_attr or 'none'} kept={row.get('kept')}",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtype", default="fp64", choices=("fp64", "fp32"))
    ap.add_argument(
        "--workloads",
        default=None,
        help="comma-separated workload names (default: all 68)",
    )
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--t", type=int, default=8)
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print(
            "attr_elementwise_probe: CUDA is required",
            file=__import__("sys").stderr,
        )
        return 1
    # The clamp widening is shipped — assert the alphabet carries it
    # so a future narrowing fails loudly here instead of silently
    # re-measuring the old exclusion.
    missing = CLAMP_OPS - gk.ELEMENTWISE
    if missing:
        print(
            f"attr_elementwise_probe: ELEMENTWISE lost {sorted(missing)}",
            file=__import__("sys").stderr,
        )
        return 1
    workloads = (
        {w.strip() for w in args.workloads.split(",") if w.strip()}
        if args.workloads
        else None
    )
    rows = measure(
        dtype=args.dtype,
        probe=args.probe,
        d=args.d,
        t=args.t,
        b=args.b,
        workloads=workloads,
    )
    n_attr = sum(1 for r in rows if r.get("attr_specs"))
    n_kept_attr = sum(1 for r in rows if r.get("attr_specs_kept"))
    print(
        f"\n{len(rows)} workloads: {n_attr} minted attr-bearing specs "
        f"across {sum(len(r.get('attr_specs', {})) for r in rows)} "
        f"claims; {n_kept_attr} workloads kept an attr claim"
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(rows, indent=1, default=str) + "\n"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
