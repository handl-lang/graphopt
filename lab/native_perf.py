"""Native-vs-Python timings: extraction, ``dag_cost``, contraction.

Reports wall-clock for the Rust walk vs the reference implementations
on representative sizes — saturated e-graphs (assoc-matmul Catalan
closure + the symmetry bank on a residual MLP) for ``extract_best``,
the extracted terms for ``dag_cost``, and seeded bond networks for
``greedy``/``cost_of_order``.  Requires the wheel
(``maturin develop --release`` in ``packages/mlopt-native``).

Usage::

    .venv/bin/python lab/native_perf.py
"""

from __future__ import annotations

import time
from typing import Any

from mlopt_core.cost import dag_cost, flops_cost
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op, Param, TensorType, Var
from mlopt_core.laws.tensor import (
    ASSOC_ADD,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    COMM_ADD,
    COMM_MUL,
)

try:
    from mlopt_native import NativeEngine
    from mlopt_native import contraction as ncon
    from mlopt_native import dag_cost as nat_dag_cost
except ModuleNotFoundError:  # pragma: no cover
    raise SystemExit(
        "mlopt_native not built — maturin develop --release "
        "in packages/mlopt-native"
    ) from None

from mlopt_torch.contraction_policy import (
    cost_of_order,
    random_bond_network,
)

__all__ = ["main"]


def _chain(k: int) -> Any:
    ws = [Param(f"W{i}", TensorType((8, 8))) for i in range(k)]
    t = ws[0]
    for w in ws[1:]:
        t = Op.make("matmul", t, w)
    return t


def _residual() -> Any:
    x = Var("x", TensorType((4, 64)))
    w1 = Param("w1", TensorType((128, 64)))
    w2 = Param("w2", TensorType((64, 128)))
    c = Param("c", TensorType((64,)))
    return Op.make(
        "add",
        x,
        Op.make(
            "linear",
            Op.make(
                "mul", Op.make("gelu", Op.make("linear", x, w1)), c
            ),
            w2,
        ),
    )


def _time(fn, reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def _extract_arm(term: Any, rules: list, label: str, reps: int) -> None:
    """Saturate once per engine, then time repeated extraction."""
    py = EGraph()
    pr = py.add_term(term)
    py.run(rules, pr, max_iterations=40, max_nodes=500_000)
    nat = NativeEngine()
    nr = nat.add_term(term)
    nat.run(rules, nr, max_iterations=40, max_nodes=500_000)
    pt = py.extract_best(pr, flops_cost)
    nt = nat.extract_best(nr, flops_cost)
    assert pt is not None and nt is not None
    t_py = _time(lambda: py.extract_best(pr, flops_cost), reps)
    t_nat = _time(lambda: nat.extract_best(nr, flops_cost), reps)
    print(
        f"extract_best {label:<22} enodes={nat.n_enodes:<7} "
        f"py={t_py * 1e3:8.3f}ms nat={t_nat * 1e3:8.3f}ms  {t_py / t_nat:5.2f}x"
    )
    # dag_cost on the extracted term.
    tp = _time(lambda: dag_cost(pt, flops_cost), reps)
    tn = _time(lambda: nat_dag_cost(nt, flops_cost), reps)
    print(
        f"{'':<4}dag_cost   {label:<22} {'':<15} "
        f"py={tp * 1e6:8.1f}us nat={tn * 1e6:8.1f}us  {tp / tn:5.2f}x"
    )


def _contraction_arm() -> None:
    for n in (20, 40, 60):
        tensors, sizes = random_bond_network(n, seed=n)

        def ref_greedy(ts_in, sz):
            def pc(x, y, s):
                c = 1.0
                for i in x | y:
                    c *= s[i]
                return c

            def mt(ts, a, b):
                m = ts[a] ^ ts[b]
                return [
                    t for k, t in enumerate(ts) if k not in (a, b)
                ] + [m]

            ts = [frozenset(t) for t in ts_in]
            total = 0.0
            while len(ts) > 1:
                cands = sorted(
                    (pc(ts[a], ts[b], sz), a, b)
                    for a in range(len(ts))
                    for b in range(a + 1, len(ts))
                )
                c, a, b = cands[0]
                total += c
                ts = mt(ts, a, b)
            return total

        reps = 20
        t_py = _time(lambda t=tensors, s=sizes: ref_greedy(t, s), reps)
        t_nat = _time(
            lambda t=tensors, s=sizes: ncon.greedy(t, s), reps
        )
        order, tot = ncon.greedy_order(tensors, sizes)
        assert tot == ref_greedy(tensors, sizes)
        t_py_co = _time(
            lambda t=tensors, s=sizes, o=order: cost_of_order(t, s, o),
            reps,
        )
        t_nat_co = _time(
            lambda t=tensors, s=sizes, o=order: ncon.cost_of_order(
                t, s, o
            ),
            reps,
        )
        print(
            f"greedy n={n:<3} py={t_py * 1e3:8.3f}ms "
            f"nat={t_nat * 1e3:8.3f}ms  {t_py / t_nat:5.2f}x   "
            f"cost_of_order {t_py_co / t_nat_co:5.2f}x"
        )


def main() -> int:
    """Print the timing table."""
    print("== extraction + dag_cost ==")
    _extract_arm(
        _chain(11),
        [ASSOC_MATMUL, ASSOC_MATMUL_REV],
        "matmul chain k=11",
        30,
    )
    _extract_arm(
        _residual(),
        [COMM_ADD, ASSOC_ADD, COMM_MUL, ASSOC_MATMUL, ASSOC_MATMUL_REV],
        "residual MLP",
        200,
    )
    print("== contraction ==")
    _contraction_arm()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
