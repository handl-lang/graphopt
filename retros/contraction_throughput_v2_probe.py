"""Per-decision cost profile + throughput/quality acceptance harness.

``project/retros/contraction-throughput-v2.md`` probe.  Two parts:

* ``--profile`` — replicates the pre-change ``rollout_orders`` inner
  loop (per-game ``ContractionGame`` + ``batch_inputs`` +
  ``Categorical``) with per-phase timers, decomposing one decision
  step into feature assembly / tensor conversion / forward / sample /
  per-game state update, at n = 20/30/40 on CPU and CUDA.
* ``--ladder`` — the honest before/after: an explicit replica of the
  old per-game rollout driver (``_old_rollouts``) against the shipped
  :func:`rollout_orders`, throughput (episodes/s at a batch sweep)
  and equal-wall-clock quality (best-of-N under an affine batch
  scheduler, the protocol of ``tools/contraction_guided_restart.py``)
  vs ``opt_einsum``'s players — every order scored by
  ``oe.contract_path``, the independent metric.

Run::

    .venv/bin/python project/retros/contraction_throughput_v2_probe.py \
        --profile
    .venv/bin/python project/retros/contraction_throughput_v2_probe.py \
        --ladder --device cuda --boards 4
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical

sys.path.insert(0, "tools")
import contraction_einsum as ce  # noqa: E402
import contraction_guided_restart as cgr  # noqa: E402
import contraction_scale as cs  # noqa: E402

from catopt_torch.contraction_policy import (  # noqa: E402
    MAX_BATCH,
    ContractionGame,
    batch_inputs,
    greedy,
    load_contraction_policy,
    pair_logits,
    random_bond_network,
    rollout_orders,
)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
#  The pre-change driver, kept explicitly for the honest before/after
# ---------------------------------------------------------------------------


def _old_rollouts(
    model: Any,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    samples: int,
    device: str,
    temperature: float,
) -> tuple[list[list[tuple[int, int]]], list[float]]:
    """The shipped pre-change lockstep driver (per-game Python state).

    This is exactly what ``rollout_orders`` did before
    ``contraction-throughput-v2``: ``samples``
    :class:`ContractionGame` clones stepped in lockstep, per-game
    feature assembly, one forward pass per step, ``.tolist()`` back to
    the host each step.  Kept inline so the before/after comparison
    runs in one process on the same boards.
    """
    games = [
        ContractionGame(tensors, sizes, greedy_ref)
        for _ in range(samples)
    ]
    orders: list[list[tuple[int, int]]] = [[] for _ in range(samples)]
    with torch.no_grad():
        while not games[0].done:
            sf, pf = batch_inputs(games, device)
            logits = pair_logits(model, sf, pf)
            idx = Categorical(logits=logits / temperature).sample()
            for j, (g, i) in enumerate(
                zip(games, idx.tolist(), strict=True)
            ):
                a, b = g.pairs[i]
                orders[j].append((a, b))
                g.step(a, b)
    return orders, [g.cost for g in games]


def _profile_rollout(
    model: Any,
    tensors: Any,
    sizes: dict[int, int],
    ref: float,
    batch: int,
    device: str,
    temperature: float = 1.0,
) -> dict[str, float]:
    """One old-driver rollout of ``batch`` episodes; per-phase seconds."""
    acc = {
        "init": 0.0,
        "feat": 0.0,
        "stack": 0.0,
        "tensor": 0.0,
        "fwd": 0.0,
        "sample": 0.0,
        "tolist": 0.0,
        "step": 0.0,
        "pairs": 0.0,
        "steps": 0,
    }
    t0 = time.perf_counter()
    games = [
        ContractionGame(tensors, sizes, ref) for _ in range(batch)
    ]
    acc["init"] = time.perf_counter() - t0
    with torch.no_grad():
        while not games[0].done:
            t0 = time.perf_counter()
            feats = [g.pair_feature_matrix() for g in games]
            sf_l = [g.state_features() for g in games]
            _sync(device)
            acc["feat"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            sf_np = np.stack(sf_l)
            pf_np = np.stack(feats)
            acc["stack"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            sf = torch.as_tensor(
                sf_np, dtype=torch.float32, device=device
            )
            pf = torch.as_tensor(
                pf_np, dtype=torch.float32, device=device
            )
            _sync(device)
            acc["tensor"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            logits = pair_logits(model, sf, pf)
            _sync(device)
            acc["fwd"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            idx = Categorical(logits=logits / temperature).sample()
            _sync(device)
            acc["sample"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            idxs = idx.tolist()
            acc["tolist"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            chosen = [g.pairs[i] for g, i in zip(games, idxs)]
            acc["pairs"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            for g, (a, b) in zip(games, chosen):
                g.step(a, b)
            acc["step"] += time.perf_counter() - t0
            acc["steps"] += 1
    return acc


def profile_section(model: Any) -> None:
    """Print the per-phase profile of the old driver on both devices."""
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        model = model.to(device)
        print(f"\n===== device {device} (old driver) =====")
        for n in (20, 30, 40):
            tensors, sizes = random_bond_network(n, 0)
            ref = greedy(tensors, sizes)
            for batch in (8, 64, 128):
                acc = _profile_rollout(
                    model, tensors, sizes, ref, batch, device
                )
                total = sum(v for k, v in acc.items() if k != "steps")
                print(
                    f"n={n:>2} B={batch:>3} "
                    f"total={1e3 * total:8.1f}ms "
                    f"ep/s={batch / total:6.0f} | "
                    f"init={1e3 * acc['init']:6.1f} "
                    f"feat={1e3 * acc['feat']:6.1f} "
                    f"stack={1e3 * acc['stack']:5.1f} "
                    f"tensor={1e3 * acc['tensor']:5.1f} "
                    f"fwd={1e3 * acc['fwd']:5.1f} "
                    f"samp={1e3 * acc['sample']:5.1f} "
                    f"list={1e3 * acc['tolist']:5.1f} "
                    f"step={1e3 * acc['step']:6.1f}"
                )


# ---------------------------------------------------------------------------
#  Throughput: episodes/s, old driver vs shipped vectorised driver
# ---------------------------------------------------------------------------


def _wall(
    fn: Any,
    args: tuple,
    reps: int = 3,
) -> float:
    """Min wall seconds of ``fn(*args)`` over ``reps`` (+1 warm)."""
    fn(*args)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def throughput_section(model: Any, device: str) -> None:
    """Print episodes/s for both drivers over a batch sweep."""
    print(f"\n== throughput, device={device} ==")
    print(
        f"  {'n':>3} {'B':>4} {'old ms':>9} {'new ms':>8} "
        f"{'old ep/s':>9} {'new ep/s':>9} {'x':>6}"
    )
    for n in (20, 30, 40):
        tensors, sizes = random_bond_network(n, 0)
        ref = greedy(tensors, sizes)
        for batch in (8, 64, 128, 256, 512):
            wo = _wall(
                _old_rollouts,
                (model, tensors, sizes, ref, batch, device, 1.0),
            )
            wn = _wall(
                rollout_orders,
                (model, tensors, sizes, ref, batch, device, 1.0),
            )
            print(
                f"  {n:>3} {batch:>4} {1e3 * wo:>9.1f} "
                f"{1e3 * wn:>8.1f} {batch / wo:>9.0f} "
                f"{batch / wn:>9.0f} {wo / wn:>6.2f}"
            )


# ---------------------------------------------------------------------------
#  Equal wall-clock quality: affine-scheduled best-of-N vs oe
# ---------------------------------------------------------------------------


def _affine_prior(
    rollouts_fn: Any,
    model: Any,
    tensors: Any,
    sizes: dict[int, int],
    ref: float,
    device: str,
) -> tuple[float, float]:
    """Fit ``f + m * B`` for one rollout driver on this board.

    Probes at (8, 256): the vectorised driver's cost is nearly flat in
    ``B`` at small batches (kernel-dispatch-bound) and grows once the
    per-step kernels saturate, so a small-batch fit underestimates the
    marginal and oversizes the first batch into the starved budgets.
    The large-batch slope is the conservative marginal.
    """
    rollouts_fn(model, tensors, sizes, ref, 4, device, 1.0)
    t0 = time.perf_counter()
    rollouts_fn(model, tensors, sizes, ref, 8, device, 1.0)
    t8 = time.perf_counter() - t0
    t0 = time.perf_counter()
    rollouts_fn(model, tensors, sizes, ref, 256, device, 1.0)
    t256 = time.perf_counter() - t0
    m = max((t256 - t8) / 248.0, 1e-6)
    f = max(t8 - m * 8, 0.0)
    return f, m


def _affine_fit(
    obs: list[tuple[int, float]], f0: float, m0: float
) -> tuple[float, float]:
    """Least-squares refit of ``f + m * B`` on observed batches."""
    bs = [b for b, _t in obs]
    if len(set(bs)) < 2:
        return f0, m0
    mb = statistics.fmean(bs)
    mt = statistics.fmean(t for _b, t in obs)
    var = statistics.fmean([(b - mb) ** 2 for b in bs])
    if var <= 0:
        return f0, m0
    m = statistics.fmean([(b - mb) * (t - mt) for b, t in obs]) / var
    f = mt - m * mb
    if m <= 0 or f < 0:
        return f0, m0
    return f, m


def _budgeted(
    rollouts_fn: Any,
    model: Any,
    tensors: Any,
    sizes: dict[int, int],
    ref: float,
    budget: float,
    device: str,
    f0: float,
    m0: float,
    seed: int,
) -> tuple[list[tuple[int, int]], float, int]:
    """Best-of-N sampled rollouts within ``budget`` (affine sizing)."""
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    best: list[tuple[int, int]] | None = None
    best_cost = float("inf")
    rollouts = 0
    obs: list[tuple[int, float]] = []
    while True:
        remaining = budget - (time.perf_counter() - t0)
        f, m = _affine_fit(obs, f0, m0)
        if remaining >= f + m:
            batch = min(MAX_BATCH, max(1, int((remaining - f) / m)))
        elif rollouts == 0:
            batch = 1
        else:
            break
        tb = time.perf_counter()
        orders, costs = rollouts_fn(
            model, tensors, sizes, ref, batch, device, 1.0
        )
        obs.append((batch, time.perf_counter() - tb))
        rollouts += batch
        k = min(range(len(costs)), key=costs.__getitem__)
        if costs[k] < best_cost:
            best, best_cost = orders[k], costs[k]
    assert best is not None
    return best, time.perf_counter() - t0, rollouts


def ladder_section(
    policy: Any,
    device: str,
    boards: int,
    budgets_ms: tuple[float, ...],
) -> None:
    """Equal-wall-clock grid: both drivers vs oe-greedy/rand-greedy."""
    print(f"\n== quality ladder, device={device}, {boards} boards ==")
    print("  oe-cost ratio vs oe-rand-greedy (mean; <1 = player wins)")
    print(
        f"  {'ms':>6} {'n':>3} {'player':>14} {'/oe-rg':>9} "
        f"{'/oe-g':>9} {'ep':>7} {'ep/s':>7} {'ms':>7}"
    )
    for n in (20, 30, 40):
        for budget in (b / 1e3 for b in budgets_ms):
            acc: dict[str, dict[str, list[float]]] = {
                p: {"r": [], "o": [], "w": [], "s": []}
                for p in ("old", "new", "oe-rg", "oe-g", "single")
            }
            for k in range(boards):
                tensors, sizes = random_bond_network(n, 100 + k)
                e = ce.to_einsum(tensors, sizes)
                ref = cs.greedy(tensors, sizes)
                # Per-board affine priors: the driver's batch cost is
                # board-dependent, and a scale-level prior underfits
                # enough to overshoot starved budgets several-fold.
                fo, mo = _affine_prior(
                    _old_rollouts,
                    policy.model,
                    tensors,
                    sizes,
                    ref,
                    device,
                )
                fn_, mn_ = _affine_prior(
                    rollout_orders,
                    policy.model,
                    tensors,
                    sizes,
                    ref,
                    device,
                )
                torch.manual_seed(0)
                orders, _c = rollout_orders(
                    policy.model,
                    tensors,
                    sizes,
                    ref,
                    1,
                    device,
                    1.0,
                    greedy=True,
                )
                single = orders[0]
                oe_single = ce.oe_cost_of_order(e, single)
                og = ce.oe_greedy_order(e)
                oe_g = ce.oe_cost_of_order(e, og)
                oe_rg, reps = cgr._oe_timed(e, budget)
                oe_r = ce.oe_cost_of_order(e, oe_rg)
                torch.manual_seed(k)
                bo, so, wo = _budgeted(
                    _old_rollouts,
                    policy.model,
                    tensors,
                    sizes,
                    ref,
                    budget,
                    device,
                    fo,
                    mo,
                    k,
                )
                torch.manual_seed(k)
                bn, sn, wn = _budgeted(
                    rollout_orders,
                    policy.model,
                    tensors,
                    sizes,
                    ref,
                    budget,
                    device,
                    fn_,
                    mn_,
                    k,
                )
                oe_bo = ce.oe_cost_of_order(e, bo)
                oe_bn = ce.oe_cost_of_order(e, bn)
                rows = {
                    "old": (oe_bo / oe_r, oe_bo / oe_g, wo, so),
                    "new": (oe_bn / oe_r, oe_bn / oe_g, wn, sn),
                    "oe-rg": (1.0, oe_r / oe_g, reps, budget),
                    "oe-g": (oe_g / oe_r, 1.0, 1, budget),
                    "single": (
                        oe_single / oe_r,
                        oe_single / oe_g,
                        1,
                        budget,
                    ),
                }
                for p, (rr, rg, w, s) in rows.items():
                    acc[p]["r"].append(rr)
                    acc[p]["o"].append(rg)
                    acc[p]["w"].append(w)
                    acc[p]["s"].append(s)
            for p in ("old", "new", "oe-rg", "oe-g", "single"):
                rr = statistics.fmean(acc[p]["r"])
                rg = statistics.fmean(acc[p]["o"])
                w = statistics.fmean(acc[p]["w"])
                s = statistics.fmean(acc[p]["s"])
                print(
                    f"  {1e3 * budget:>6.0f} {n:>3} {p:>14} "
                    f"{rr:>9.3f} {rg:>9.3f} {w:>7.1f} "
                    f"{w / s:>7.0f} {1e3 * s:>7.1f}"
                )


def main() -> None:
    """Parse args and run the requested sections."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boards", type=int, default=4)
    ap.add_argument("--budgets", default="50,200,1000")
    args = ap.parse_args()
    if not (args.profile or args.throughput or args.ladder):
        args.profile = args.throughput = args.ladder = True
    policy = load_contraction_policy(device=args.device)
    budgets = tuple(
        float(x) for x in args.budgets.split(",") if x.strip()
    )
    if args.profile:
        profile_section(policy.model)
    if args.throughput:
        for dev in ("cpu", args.device):
            if dev == "cuda" and not torch.cuda.is_available():
                continue
            throughput_section(policy.model.to(dev), dev)
    if args.ladder:
        ladder_section(policy, args.device, args.boards, budgets)


if __name__ == "__main__":
    main()
