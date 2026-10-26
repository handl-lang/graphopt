"""Does a **real kernel** make a prim fold pay?

`docs/reports/pay_measured.md` measured the shipped ``synop*`` family
losing on wall clock and named the cause: the denotation is a Python
spec-walk, not a fused kernel.  The `gen` domain shows the other half —
a *generated* Triton kernel beats `torch.compile` by 2.87x on a 6-op
chain.  This probe asks the only question left: **if the fold's body were
a kernel, would the fold pay?**

It is an **op-level** measurement, not a module-level one, and that is
deliberate: the delivery probe times fp64 on CPU, where a Triton kernel
cannot run at all.  Here every arm is CUDA fp32 on a realistic element
count, and the three arms are:

* ``binding`` — the shipped denotation (``shipped_primitives`` spec-walk),
  i.e. what a ``synop`` call costs today;
* ``kernel`` — a generated ``triton.jit`` body for the same spec
  (``genkernel.triton_bindings``), verified against the binding first;
* ``eager`` — the manual spelling evaluated with plain torch calls: the
  floor a fused body has to beat.

A kernel that does not match the binding on the same inputs is **refused,
not measured**.  Ops whose spec is not elementwise (or that the codegen
does not model) are declined with the reason — the family is not uniform.

Run::

    .venv/bin/python -m bench.drivers.probes.prim_kernel_probe
    .venv/bin/python -m bench.drivers.probes.prim_kernel_probe --json /tmp/k.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from mlopt_core.laws import minted
from mlopt_discovery import genkernel
from mlopt_discovery import shipped_primitives as sp

DEFAULT_OPS = (
    "synop_bb8c54c201",
    "synop_5e58f26875",
    "synop_4c657cb2c0",
    "synop_2b8cfd1c90",
)
ELEMS = 2048 * 512  # 2-D: elementwise bodies, one flat index space


def elementwise_ops(limit: int = 6) -> list[str]:
    """The shipped prims whose denotation is a pointwise body.

    Only these can receive a generated elementwise kernel; the rest stay
    on their torch path and are reported as declined, not as losses.
    """
    out: list[str] = []
    for op in sorted(minted.SYNOP_SHIPPED):
        try:
            if genkernel.elementwise_spec(sp.spec(op)):
                out.append(op)
        except Exception:
            continue
        if len(out) >= limit:
            break
    return out


def kernels_for(
    ops: list[str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Return ``{op: callable}`` for the ops a Triton body can cover.

    The callable is the generated kernel; ops the codegen does not model
    come back as notes, not as binds, so a caller can bind the rest
    without pretending coverage.
    """
    binds: dict[str, Any] = {}
    notes: list[dict[str, str]] = []
    for op in ops:
        arms, why = _arms(op)
        if why or "kernel" not in arms:
            notes.append(
                {
                    "op": op,
                    "status": "declined",
                    "why": why or "no codegen",
                }
            )
            continue
        binds[op] = arms["kernel"]
        notes.append({"op": op, "status": "bound", "why": "generated"})
    return binds, notes


def _arms(op: str) -> tuple[dict[str, Any], str | None]:
    """Build the three arms for *op*; return (arms, decline reason)."""

    fold, _expand = minted.prim_fold(op)
    spec = sp.spec(op)
    n_args = len(fold.rhs.args)
    if n_args == 0:
        return {}, "no arguments"
    metas = tuple(fold.rhs.args)
    attrs = tuple(fold.rhs.attrs)
    binding = sp._binding(spec, metas, attrs)

    arms: dict[str, Any] = {"binding": binding}
    if genkernel.elementwise_spec(spec):
        handler = {
            f"{op}_k": {
                "pattern": spec,
                "kernel": f"{op}_k",
                "args": tuple(genkernel._spec_metas(spec)),
            }
        }
        made = genkernel.triton_bindings(handler)
        fn = next(iter(made.values()), None)
        if fn is not None:
            arms["kernel"] = fn
    return arms, None


def _inputs(
    n_args: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[Any, ...]:
    """Realistic inputs — the kernel tier is CUDA-only."""
    g = torch.Generator(device="cpu").manual_seed(0)
    base = (torch.rand(ELEMS, generator=g) + 0.5).to(dtype)
    return tuple(
        (base if i % 2 == 0 else base * 2.0)
        .reshape(2048, 512)
        .to(device)
        for i in range(n_args)
    )


def _time(fn: Any, args: tuple[Any, ...], iters: int = 200) -> float:
    """Median ms over *iters* calls, after a short warmup."""
    with torch.inference_mode():
        for _ in range(20):
            fn(*args)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(*args)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def measure(
    op: str, device: torch.device, dtype: torch.dtype = torch.float32
) -> dict[str, Any]:
    """Time the arms for one op and report the ratios."""
    arms, why = _arms(op)
    if why:
        return {"op": op, "status": "declined", "why": why}
    fold, _ = minted.prim_fold(op)
    try:
        args = _inputs(len(fold.rhs.args), device, dtype)
    except Exception as exc:
        return {
            "op": op,
            "status": "declined",
            "why": f"inputs: {type(exc).__name__}: {exc}",
        }
    with torch.inference_mode():
        want = arms["binding"](*args)
    row: dict[str, Any] = {
        "op": op,
        "status": "measured",
        "ms": {
            name: round(_time(fn, args), 5) for name, fn in arms.items()
        },
    }
    if "kernel" in arms:
        with torch.inference_mode():
            got = arms["kernel"](*args)
        ok = bool(torch.allclose(want, got, rtol=1e-3, atol=1e-4))
        row["kernel_matches_binding"] = ok
        if not ok:
            row["status"] = "refused"
            row["why"] = (
                "kernel mismatch max_abs="
                f"{float((want - got).abs().max()):.3e}"
            )
            row["ms"].pop("kernel", None)
        else:
            row["kernel_over_binding"] = round(
                row["ms"]["kernel"] / row["ms"]["binding"], 3
            )

    else:
        row["status"] = "declined"
        row["why"] = "no Triton codegen for this spec"
    return row


def run(
    ops: list[str], device: str = "cuda", dtype: str = "float32"
) -> dict[str, Any]:
    """Measure every requested op."""
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    dt = torch.float64 if dtype == "float64" else torch.float32
    rows = [measure(op, dev, dt) for op in ops]
    return {
        "device": str(dev),
        "dtype": dtype,
        "elements": ELEMS,
        "rows": rows,
    }


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value}"


def summary(result: dict[str, Any]) -> str:
    """Render the per-op arms and ratios."""
    lines = [
        f"device {result['device']} · {result['elements']} elements · "
        f"{result.get('dtype', 'float32')}",
        "",
        "| op | status | binding ms | kernel ms | kernel/binding |",
        "|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        ms = row.get("ms", {})
        lines.append(
            f"| `{row['op']}` | {row['status']} | "
            f"{_fmt(ms.get('binding'))} | {_fmt(ms.get('kernel'))} | "
            f"{_fmt(row.get('kernel_over_binding'))} |"
        )
    notes = [r for r in result["rows"] if r.get("why")]
    if notes:
        lines += ["", "declined / refused:"]
        lines += [f"- `{r['op']}`: {r['why']}" for r in notes]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the probe and print the table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ops",
        nargs="*",
        default=None,
        help="default: the shipped pointwise prims",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--dtype", default="float32", choices=["float32", "float64"]
    )
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args(argv)
    ops = args.ops if args.ops else elementwise_ops()
    result = run(ops, args.device, args.dtype)
    print(summary(result))
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
