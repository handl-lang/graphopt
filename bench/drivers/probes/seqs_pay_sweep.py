"""seqs pay sweep — does the emitted-scan tier beat ``torch.cumsum``?

Standalone probe behind ``docs/reports/seqs_pay_sweep.md``.  Times
the generated ``kind="seqs"`` Triton binding (``_gs`` strided /
``_gsc`` dense) against the spelled incumbent ``torch.cumsum`` over
a (T, N) grid across the operand layouts the zoo sites actually
mint on, plus the dense-fallback and fold-mode arms and two
in-module ``_zoo_case`` referee runs.

Serial GPU probe — small batches, modest reps.  Run::

    .venv/bin/python bench/drivers/probes/seqs_pay_sweep.py
"""

from __future__ import annotations

import argparse
import sys

import torch
from mlopt_discovery import genkernel as gk

_CUMSPEC = (
    "applyd",
    ("aff_diag", 1, "X1"),
    "H",
    {"kind": "seqs", "axis": 1},
)
_FOLD_SPEC = (
    "applyd",
    ("aff_diag", ("mul", "X1", 2.0), "X1"),
    "H",
    {"kind": "seqs", "axis": 1},
)


def _binding(spec, tag, axis):
    """One seqs binding callable for *spec* (axis pinned in handler)."""
    h = {
        "pattern": spec,
        "kernel": tag,
        "args": ("X1", "H"),
        "axis": axis,
    }
    out = gk.scan_triton_bindings({tag: h})
    return out[tag]


def _time_pair(spelled, kernel, reps):
    """Interleaved median window; returns (t_spelled_us, t_kernel_us)."""
    ts, tk = gk._paired_time(spelled, kernel, reps)
    return ts * 1e6, tk * 1e6


def _row(tag, shape, axis, dtype, reps, note=""):
    """Time one operand shape; print a sweep row."""
    kern = _binding(_CUMSPEC, f"sw_{tag}", axis)
    x = torch.randn(
        *shape,
        dtype=dtype,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(0),
    )
    h = torch.zeros((), dtype=dtype, device="cuda")
    # verify once — the referee's verify-before-time gate
    try:
        got = kern(x, h)
        ref = torch.cumsum(x, dim=axis)
        ok = torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
    except Exception as exc:
        print(
            f"{tag:>10} shape={tuple(shape)!s:>18} ax={axis} "
            f"{str(dtype).removeprefix('torch.'):>5} "
            f"kernel=RAISE {type(exc).__name__}: {exc}"
        )
        return None, None
    spelled = lambda: torch.cumsum(x, dim=axis)  # noqa: E731
    tkern = lambda: kern(x, h)  # noqa: E731
    ts, tk = _time_pair(spelled, tkern, reps)
    print(
        f"{tag:>10} shape={tuple(shape)!s:>18} ax={axis} "
        f"{str(dtype).removeprefix('torch.'):>5} "
        f"spelled={ts:8.1f}us kernel={tk:8.1f}us "
        f"ratio={tk / ts:6.3f} verify={'ok' if ok else 'FAIL'} {note}"
    )
    return ts, tk


def _decline_row(tag, shape, axis, dtype):
    """Record the T>256 honest decline; time the incumbent only."""
    kern = _binding(_CUMSPEC, f"sw_{tag}", axis)
    x = torch.randn(
        *shape,
        dtype=dtype,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(0),
    )
    h = torch.zeros((), dtype=dtype, device="cuda")
    try:
        kern(x, h)
        outcome = "ran?!"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    spelled = lambda: torch.cumsum(x, dim=axis)  # noqa: E731
    ts = gk._median_time(spelled, 20) * 1e6
    print(
        f"{tag:>10} shape={tuple(shape)!s:>18} ax={axis} "
        f"{str(dtype).removeprefix('torch.'):>5} "
        f"kernel=DECLINE ({outcome}) spelled={ts:8.1f}us"
    )


def sweep_strided(reps):
    """The ``_gs`` strided path — layouts the zoo sites mint on."""
    print("== strided _gs: 3-D (B,T,N) axis=1, B=8, fp64 ==")
    for n in (64, 256, 1024):
        for t in (32, 64, 128, 256):
            _row("3d", (8, t, n), 1, torch.float64, reps)
    print("== strided _gs: 2-D (T,N) axis=0, fp64 ==")
    for n in (64, 256, 1024):
        for t in (64, 256):
            _row("2d", (t, n), 0, torch.float64, reps)
    print("== strided _gs: 4-D zoo-realistic (B,T,d,d) axis=1, fp64 ==")
    for d in (8, 16, 32):  # K = d*d = 64/256/1024
        for t in (32, 64, 128, 256):
            _row("4d", (8, t, d, d), 1, torch.float64, reps)


def sweep_declines(reps):
    """T > _SEQS_MAX_T — the closed-form kernel honestly raises."""
    print("== T>256 declines (3-D (B,T,N) axis=1, fp64) ==")
    for t in (512, 1024, 4096, 16384):
        _decline_row("3d-bigT", (8, t, 256), 1, torch.float64)
    print("== T>256 declines (4-D (B,T,d,d) axis=1, fp64) ==")
    for t in (1024, 4096):
        _decline_row("4d-bigT", (8, t, 16, 16), 1, torch.float64)


def sweep_dense(reps):
    """Non-contiguous operand -> the ``_gsc`` dense ``(T,N)`` path."""
    print("== dense _gsc fallback (non-contiguous (B,T,N)) ==")
    for n in (256, 1024):
        for t in (64, 256):
            base = torch.randn(
                8,
                n,
                t,
                dtype=torch.float64,
                device="cuda",
                generator=torch.Generator(device="cuda").manual_seed(0),
            )
            x = base.permute(0, 2, 1)  # (8,T,N), stride-forced dense
            kern = _binding(_CUMSPEC, f"sw_d{n}_{t}", 1)
            h = torch.zeros((), dtype=torch.float64, device="cuda")
            got = kern(x, h)
            ref = torch.cumsum(x, dim=1)
            ok = torch.allclose(got, ref, rtol=1e-9, atol=1e-9)
            spelled = lambda x=x: torch.cumsum(x, dim=1)  # noqa: E731
            tkern = lambda x=x, h=h, kern=kern: kern(x, h)  # noqa: E731
            ts, tk = _time_pair(spelled, tkern, reps)
            print(
                f"{'dense':>10} shape={(8, t, n)!s:>18} ax=1 fp64 "
                f"spelled={ts:8.1f}us kernel={tk:8.1f}us "
                f"ratio={tk / ts:6.3f} verify={'ok' if ok else 'FAIL'}"
            )


def sweep_fold(reps):
    """``scan_mode='fold'`` — the serial emitted fold, any T."""
    print("== fold mode (serial emitted fold) ==")
    for t, n in ((256, 256), (1024, 64)):
        x = torch.randn(
            8,
            t,
            n,
            dtype=torch.float64,
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
        h = torch.zeros((), dtype=torch.float64, device="cuda")
        kern = _binding(_FOLD_SPEC, f"sw_f{t}_{n}", 1)
        got = kern(x, h)
        # spelled fold denotation (the referee's arm for fold mode is
        # the generic emitted fold, not torch.cumsum)
        acc = torch.zeros(8, n, dtype=torch.float64, device="cuda")
        outs = []
        for i in range(t):
            acc = (2.0 * x[:, i]) * acc + x[:, i]
            outs.append(acc)
        ref = torch.stack(outs, dim=1)
        ok = torch.allclose(got, ref, rtol=1e-8, atol=1e-8)
        spelled = lambda x=x: torch.cumsum(x, dim=1)  # noqa: E731
        tkern = lambda x=x, h=h, kern=kern: kern(x, h)  # noqa: E731
        ts, tk = _time_pair(spelled, tkern, reps)
        print(
            f"{'fold':>10} shape={(8, t, n)!s:>18} ax=1 fp64 "
            f"vs_cumsum spelled={ts:8.1f}us kernel={tk:8.1f}us "
            f"ratio={tk / ts:6.3f} verify={'ok' if ok else 'FAIL'}"
        )


def sweep_fp32(reps):
    """fp32 spot checks — the dtype a caller actually ships."""
    print("== fp32 spot checks (3-D (B,T,N) axis=1) ==")
    for t, n in ((64, 256), (256, 1024), (256, 64), (128, 1024)):
        _row("3d-f32", (8, t, n), 1, torch.float32, reps)


def sweep_bn(reps):
    """``_SEQS_BN`` sensitivity at two shapes (module-global patch)."""
    print("== BN sensitivity (3-D (B,T,N) axis=1, fp64) ==")
    orig = gk._SEQS_BN
    try:
        for bn in (128, 512):
            gk._SEQS_BN = bn  # ty: ignore[invalid-assignment]
            for t, n in ((256, 1024), (64, 4096)):
                _row(
                    f"bn{bn}",
                    (8, t, n),
                    1,
                    torch.float64,
                    reps,
                    note=f"BN={bn}",
                )
    finally:
        gk._SEQS_BN = orig


def sweep_inmodule(_reps=30):
    """The real referee: ``play._zoo_case`` on CumsumLinearAttention."""
    from mlopt_discovery import play
    from mlopt_discovery import zoo as _zoo

    print("== in-module _zoo_case referee ==")
    for d, t, b in (
        (64, 32, 8),  # the report's measured shape (K=4096)
        (16, 64, 8),  # island candidate: K=256, T=64, B=8
        (16, 32, 8),
        (16, 128, 8),
        (64, 256, 8),  # T at the kernel cap
        (128, 64, 8),
    ):
        wls = [
            w
            for w in _zoo.zoo(d, t, b)
            if w.name == "CumsumLinearAttention"
        ]
        if not wls:
            print(f"d={d} t={t} b={b}: workload not found")
            continue
        mark = len(gk.PROFITABLE_LOG)
        case = play._zoo_case(wls[0])
        ex = case[5]
        genh = ex["gen_handlers"]
        seqs_rows = [
            r for r in gk.PROFITABLE_LOG[mark:] if "scan" in str(r[0])
        ]
        print(
            f"d={d} t={t} b={b}: gen_handlers={sorted(genh)} "
            f"seqs_referee_rows={seqs_rows}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument(
        "--only",
        default="",
        help="comma list: strided,declines,dense,fold,fp32,bn,inmodule",
    )
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        sys.exit("needs CUDA")
    arms = {
        "strided": sweep_strided,
        "declines": sweep_declines,
        "dense": sweep_dense,
        "fold": sweep_fold,
        "fp32": sweep_fp32,
        "bn": sweep_bn,
        "inmodule": sweep_inmodule,
    }
    for name, fn in arms.items():
        if only is None or name in only:
            fn(args.reps)


if __name__ == "__main__":
    main()
