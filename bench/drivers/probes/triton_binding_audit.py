"""Why does ``_triton_kernel_for`` never bind on the claim-bearing zoo?

The ``zoo_chain`` sweeps recorded that for every minted handler on the
claim-bearing workloads, in both dtypes, the referee's ``kernel_for``
hook returned no binding — so ``profitable`` has only ever graded
compiled-spelled bodies on this corpus, and "the generated kernel
beats spelled" has no measurement attached to it.

This driver runs the same shipped path as ``zoo_chain``
(``play._zoo_case`` → ``_mint_tiers`` → ``genkernel.profitable`` with
``kernel_for=_triton_kernel_for``) on every claim-bearing workload at
both intake dtypes, and dumps ``play.TRITON_REFEREE_LOG`` — the
per-consult audit trail the hook appends: which tiers were consulted,
whether each bound or declined, the rejection reason, and (via a
``_kernel_pays`` wrapper) what the referee did with each bound kernel.

Outcomes:

- ``bound`` — a generated kernel was delivered to the referee;
  ``referee`` (from the ``_kernel_pays`` wrapper) then says whether it
  verified and beat spelled (``{"spelled_us", "kernel_us", "ratio"}``),
  failed verification, or raised;
- ``declined`` — every registered builder passed; ``attempts`` says
  whether the spec was outside each tier's grammar, or recognised but
  its generated module failed to load;
- ``env_bind_failed`` — a builder bound but the CUDA env could not be
  assembled (non-tensor metavar leaf, missing binding);
- ``unavailable`` — no CUDA / no triton.

(The pre-fix run also recorded ``pointwise_would_bind`` — whether the
pointwise codegen ``hybrid_sink`` ships, which ``TIERS`` then left
unregistered, would have bound the spec the registry declined.  It was
true for all 86 declines; the field is gone now that the gen tier
registers that builder.)

Run::

    .venv/bin/python -m bench.drivers.probes.triton_binding_audit \\
        --out docs/reports/triton_binding_audit.json
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

#: The zoo workloads either ``zoo_chain`` sweep minted at least one
#: claim on (union of the fp64 and fp32 tables).
DEFAULT_ONLY = (
    "GatedAttentionUnit,ChunkedRetention,DeltaRuleBlock,SplineKAN,"
    "RWKVMixBlock,GEGLUBlock,ManualGroupNorm,InstanceNormBroadcast,"
    "GatedDeltaRule,SigmoidAttention,TaylorLinearAttention"
)


def _audited_kernel_pays(
    kernel: Any, spelled: Any, vals: dict, mvs: list, reps: int
) -> tuple | None:
    """``genkernel._kernel_pays`` with the decline reason kept.

    Same verify-then-time protocol — the wrapper annotates the
    consult's ``TRITON_REFEREE_LOG`` record so a bound claim shows
    *why* the referee dropped it: output mismatch / kernel raised
    (``verify``) vs bound-but-not-faster (``timed``).
    """
    rec = play.TRITON_REFEREE_LOG[-1] if play.TRITON_REFEREE_LOG else {}
    try:
        out = gk._referee_out(kernel, vals, mvs)
        if not torch.allclose(
            out, gk._host(spelled()), rtol=1e-4, atol=1e-5
        ):
            rec["referee"] = "verify_failed"
            return None
        t_spelled = gk._median_time(spelled, reps)
        t_kernel = gk._median_time(
            lambda: kernel(*[vals[m] for m in mvs]), reps
        )
    except Exception as exc:
        rec["referee"] = f"raised: {type(exc).__name__}: {exc}"
        return None
    rec["referee"] = {
        "spelled_us": round(t_spelled * 1e6, 1),
        "kernel_us": round(t_kernel * 1e6, 1),
        "ratio": round(t_spelled / t_kernel, 3),
    }
    return t_spelled, t_kernel


def audit(wl: Any, dtype: str) -> list[dict]:
    """Run one (workload, dtype) cell; return its referee records."""
    n0 = len(play.TRITON_REFEREE_LOG)
    orig = gk._kernel_pays
    gk._kernel_pays = _audited_kernel_pays
    try:
        case = play._zoo_case(wl, dtype=dtype)
    finally:
        gk._kernel_pays = orig
    kept = set(case[5]["gen_handlers"])
    recs = play.TRITON_REFEREE_LOG[n0:]
    for r in recs:
        r["workload"] = wl.name
        r["swept_dtype"] = dtype
        r["kept_by_referee"] = r["tag"] in kept
    return recs


def measure(
    *, d: int, t: int, b: int, only: tuple[str, ...]
) -> list[dict]:
    """Audit every requested workload at both dtypes."""
    want = set(only)
    rows: list[dict] = []
    for wl in zoo_mod.zoo(d, t, b):
        if want and wl.name not in want:
            continue
        for dtype in ("fp64", "fp32"):
            try:
                recs = audit(wl, dtype)
            except Exception as exc:
                recs = [
                    {
                        "workload": wl.name,
                        "swept_dtype": dtype,
                        "outcome": "case_failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                ]
            rows.extend(recs)
            summary = ", ".join(
                f"{r.get('tag')}: {r['outcome']}" for r in recs
            )
            print(f"zoo:{wl.name}[{dtype}]: {summary}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--t", type=int, default=32)
    ap.add_argument("--b", type=int, default=8)
    ap.add_argument("--only", default=DEFAULT_ONLY)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        print(
            "triton_binding_audit: CUDA is required",
            file=__import__("sys").stderr,
        )
        return 1
    only = tuple(x for x in args.only.split(",") if x)
    rows = measure(d=args.d, t=args.t, b=args.b, only=only)
    bound = sum(1 for r in rows if r["outcome"] == "bound")
    print(
        f"\n{len(rows)} referee consults: {bound} bound, "
        f"{sum(1 for r in rows if r['outcome'] == 'declined')} declined, "
        f"{sum(1 for r in rows if r['outcome'] == 'env_bind_failed')} "
        "env-bind failures"
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(rows, indent=1, default=str) + "\n"
        )
        print(f"wrote {args.out} ({len(rows)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
