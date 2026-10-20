"""The conv+BN fold's three wall-clock regimes — one binary, one fixture.

``bench/RESULTS.md`` cites three different pay numbers for the same
fold: -16 % via ``hnl-run --repeat`` (``tests/test_kit_bridge.py``),
-5 % deployed / +33 % folded-inline via the scaffold's
``conv-bn-fold --repeat`` bin (``research-mlopt``'s ``experiments/``
crate, in-tree on ``main``), and the stable ``dag_cost``
halving.  They are not in disagreement — ``--repeat`` re-evaluates the
*whole program* per call, so each number prices the fold math
(``channel_scales``' ``.hnl`` while-loop + the ``W'``/``b'`` tensor
updates) in a different place:

* **deployed** — ``W'``/``b'`` materialized once, *outside* the timed
  loop; per-call is a bare ``conv2d``.  The honest deployed regime.
* **folded-inline** — ``conv_bn_fold`` recomputes the fold every call:
  the un-amortized regime, a net per-call *loss*.  This is why the
  law's pay is structural — a weight-space fold is compile-time work.
* **the recorded -16 %** — the bridge test's spelled arm binds its
  whole ``feed`` dict in the program prelude (``emit_program`` binds
  every key), including the ``cs``/``s`` marshal escape the spelled
  term never reads; every ``--repeat`` evaluates ~10 ms of prelude
  the op doesn't need.  Trimming the prelude to the term's own
  leaves isolates the artifact.

This probe runs all five arms through the *same* debug ``hnl-run``
binary on the *same* fixture (the bridge test's: 1x4x24x24 input,
8-channel 1x1 conv) so the decomposition is measured, not argued —
plus ``dag_cost`` on the same terms, which is box-independent.

Run with
``.venv/bin/python bench/probes/conv_fold_wall_regimes_probe.py``
(needs ``target/debug/hnl-run``, built by
``cargo build -p mlopt-kit --bin hnl-run``; ~4-6 min on this box —
five programs x 31 evals through a debug tree-walk).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
from mlopt_core.cost.basic import flops_cost
from mlopt_core.cost.params import dag_cost
from mlopt_core.ir import Op, Param, TensorType, Var

_REPO = Path(__file__).resolve().parents[2]
_BIN = _REPO / "target" / "debug" / "hnl-run"
_PROJECT = _REPO / "packages" / "mlopt-kit"
_REPS = 30

#: The same marshaled escape ``tests/test_kit_bridge.py`` feeds: the
#: op surface has no elementwise ``rsqrt``, so the channel scales ride
#: the ``{"hnl"}`` channel — a ``while``/``match`` loop over channels.
_CHANNEL_SCALES = (
    "(gs, vs, eps) ~> while { gs: gs, vs: vs, acc: [] } {\n"
    "  (gs, vs, acc) ~> match gs {\n"
    "    []: { out: acc },\n"
    "    [g, ...gt]: match vs {\n"
    "      []: { out: acc },\n"
    "      [v, ...vt]: { state: { gs: gt,\n"
    "                            vs: vt,\n"
    "                            acc: [...acc, g / ((v + eps) ** 0.5)] } }\n"
    "    }\n"
    "  }\n"
    "}"
)


def _f(t: torch.Tensor) -> dict[str, Any]:
    """Return *t* as a ``{"shape", "data"}`` feed entry."""
    return {
        "shape": list(t.shape),
        "data": t.detach().reshape(-1).tolist(),
    }


def _hnl_tensor(t: dict[str, Any]) -> str:
    """Render a feed entry as a `.hnl` literal expression."""
    data = ", ".join(repr(float(v)) for v in t["data"])
    return (
        f"tensor.reshape(tensor.from_list([{data}]), "
        f"{list(t['shape'])})"
    )


def _per_call(run: Any, src: str, reps: int = _REPS) -> float:
    """``(t(n) - t(1)) / (n - 1)`` — the ``--repeat`` cost formula."""
    t1 = time.perf_counter()
    run(src)
    t1 = time.perf_counter() - t1
    tn = time.perf_counter()
    run(src, repeat=reps)
    tn = time.perf_counter() - tn
    return (tn - t1) / (reps - 1)


def main() -> int:
    """Time all five arms through one binary; print the account."""
    # `bench/` is not an installed package — resolve it from the repo
    # root so the script runs as documented (same shape the other
    # `bench.*`-importing probes need).
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    from bench.kit import term_hnl as kh

    kit = kh.KitRunner(_BIN)
    kit_proj = kh.KitRunner(_BIN, project=_PROJECT)

    T = TensorType
    n, c_in, c_out, hw, eps = 1, 4, 8, 24, 1e-5
    x = Var("x", T((n, c_in, hw, hw)))
    w = Param("w", T((c_out, c_in, 1, 1)))
    b, g, be, m, v, s = (
        Param(p, T((c_out,))) for p in ("b", "g", "be", "m", "v", "s")
    )
    w2 = Param("w2", T((c_out, c_in, 1, 1)))
    b2 = Param("b2", T((c_out,)))

    unfolded = Op.make(
        "batch_norm",
        Op.make("conv2d", x, w, b),
        g,
        be,
        m,
        v,
        training=False,
        momentum=0.1,
        eps=eps,
        cudnn_enabled=False,
    )
    deployed = Op.make("conv2d", x, w2, b2)
    w2_term = Op.make(
        "mul", w, Op.make("reshape", s, shape=(c_out, 1, 1, 1))
    )
    b2_term = Op.make(
        "add",
        Op.make("mul", b, s),
        Op.make("sub", be, Op.make("mul", s, m)),
    )

    torch.manual_seed(2)
    feed = {
        "x": _f(torch.randn(n, c_in, hw, hw, dtype=torch.float64)),
        "w": _f(torch.randn(c_out, c_in, 1, 1, dtype=torch.float64)),
        "b": _f(torch.randn(c_out, dtype=torch.float64)),
        "g": _f(torch.randn(c_out, dtype=torch.float64).abs() + 0.5),
        "be": _f(torch.randn(c_out, dtype=torch.float64)),
        "m": _f(torch.randn(c_out, dtype=torch.float64)),
        "v": _f(torch.rand(c_out, dtype=torch.float64) + 0.25),
        "cs": {"hnl": _CHANNEL_SCALES},
        "s": {
            "hnl": (
                "tensor.from_list(cs(tensor.to_list(g), "
                "tensor.to_list(v), 0.00001))"
            )
        },
    }

    # The compile-time half — the fold materialized once, untimed.
    folded_w = kit.run(kh.emit_program(w2_term, feed))["__tensor__"]
    folded_b = kit.run(kh.emit_program(b2_term, feed))["__tensor__"]

    print(f"== conv+BN fold wall regimes (hnl-run --repeat {_REPS}) ==")

    # Arm 1 — the recorded shape: the spelled program's prelude binds
    # the whole feed, cs/s escape included.
    t_spelled_full = _per_call(kit.run, kh.emit_program(unfolded, feed))
    print(
        f"spelled (feed incl. cs/s escape): {t_spelled_full * 1e3:8.2f} ms"
    )

    # Arm 2 — same term, prelude trimmed to its own leaves.
    trim = {k: feed[k] for k in ("x", "w", "b", "g", "be", "m", "v")}
    t_spelled_trim = _per_call(kit.run, kh.emit_program(unfolded, trim))
    print(
        f"spelled (trimmed prelude)       : {t_spelled_trim * 1e3:8.2f} ms"
    )

    # Arm 3 — deployed: conv2d over weights folded once, untimed.
    deploy_feed = {"x": feed["x"], "w2": folded_w, "b2": folded_b}
    t_deployed = _per_call(
        kit.run, kh.emit_program(deployed, deploy_feed)
    )
    print(
        f"deployed conv2d(x, W2, b2)      : {t_deployed * 1e3:8.2f} ms"
    )

    # Arms 4-5 — the scaffold's shape through the same binary: the
    # `.hnl` module exports called per eval, fold math inline or not.
    args = (
        ", ".join(
            _hnl_tensor(feed[k])
            for k in ("x", "w", "b", "g", "be", "m", "v")
        )
        + ", 0.00001"
    )
    t_mod: dict[str, float] = {}
    for export in ("conv_bn", "conv_bn_fold"):
        src = (
            f'do {{\nfrom "./hnl/conv_bn.hnl" import {export}\n'
            f"{export}({args})\n}}"
        )
        t_mod[export] = _per_call(kit_proj.run, src)
        print(f"module {export:<22}: {t_mod[export] * 1e3:8.2f} ms")

    print(
        "\n-- deltas ------------------------------------------------"
    )
    print(
        f"deployed vs spelled (full feed) : "
        f"{(t_deployed / t_spelled_full - 1) * 100:+.1f} %  "
        "<- the recorded -16% arm"
    )
    print(
        f"deployed vs spelled (trimmed)   : "
        f"{(t_deployed / t_spelled_trim - 1) * 100:+.1f} %  "
        "<- op-level, prelude isolated"
    )
    print(
        f"deployed vs module conv_bn      : "
        f"{(t_deployed / t_mod['conv_bn'] - 1) * 100:+.1f} %  "
        "<- deployed regime, scaffold shape"
    )
    print(
        f"inline fold vs module conv_bn   : "
        f"{(t_mod['conv_bn_fold'] / t_mod['conv_bn'] - 1) * 100:+.1f} %  "
        "<- un-amortized fold per call"
    )
    print(
        f"\ndag_cost spelled={dag_cost(unfolded, flops_cost):.0f} "
        f"deployed={dag_cost(deployed, flops_cost):.0f} "
        "(box-independent — the decision's stable price)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
