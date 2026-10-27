"""The deploy-regime referee — honest timing for memory-bound kernels.

A kernel measured in a warm cache is not the kernel that runs in a
deployment. On this class of card the L2 is ~2 MB and a single
transformer layer's int4 weights are 2.18 MB, so by the time a layer is
used again its weights are gone. Warm benchmarks therefore flatter every
kernel — ours and the incumbent's alike — and a search that optimizes a
warm number picks winners that lose in production.

This module provides the primitive the rest of the bench should use:

* :func:`evictor` — a cheap pass over a large buffer that pushes the
  resident weights out of L2,
* :func:`blocked_paired` — measure *k* callables with drift cancelled:
  every round visits all of them in a fresh shuffled order, warming each
  one before its timed block, so a drifting clock hits every candidate
  equally while each measurement stays in the regime a caller sees,
* :func:`roofline_us` — the bytes ÷ bandwidth ceiling, so a timing can be
  reported as a fraction of what the hardware allows.

The protocol matters: sequential measurement favours whoever ran first
(measured: 58 µs vs 81 µs on identical code), naive round-robin evicts
the weights it is measuring (116 µs), and this blocked design is the one
that reproduced a stable ranking.
"""

from __future__ import annotations

import statistics as stats
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

__all__ = [
    "REGIME_LEVELS",
    "RegimeResult",
    "blocked_paired",
    "evictor",
    "regime_curve",
    "roofline_us",
]

#: Default eviction buffer: comfortably larger than a laptop L2.
DEFAULT_EVICT_MB = 64

#: Passes that reproduce deployment pressure on this class of card.
DEPLOY_EVICT_PASSES = 4

#: The curve worth reporting: none, light, deploy-like, heavy.
REGIME_LEVELS: tuple[int, ...] = (0, 1, 4, 16)


def evictor(evict_mb: int = DEFAULT_EVICT_MB, device: str = "cuda"):
    """Return ``flush(passes)`` — a cache-evicting pass over a big buffer.

    ``passes=0`` flushes nothing (the warm-cache regime); a larger count
    removes more of the resident weight from L2. The buffer is allocated
    once and reused, so a pass costs a single elementwise kernel.
    """
    buf = torch.empty(
        max(1, evict_mb) * 1024 * 1024 // 4,
        dtype=torch.float32,
        device=device,
    )

    def flush(passes: int = 1) -> None:
        for _ in range(max(0, passes)):
            buf.mul_(1.0000001)

    return flush


def roofline_us(
    bytes_moved: float, bandwidth_gbps: float, overhead_us: float = 0.0
) -> float:
    """Lower bound on time: bytes ÷ measured bandwidth + fixed overhead."""
    if bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be positive")
    return bytes_moved / (bandwidth_gbps * 1e9) * 1e6 + overhead_us


@dataclass
class RegimeResult:
    """Per-callable timings from one blocked-randomized run."""

    medians: dict[str, float] = field(default_factory=dict)
    stdevs: dict[str, float] = field(default_factory=dict)
    samples: dict[str, list[float]] = field(default_factory=dict)
    evict_passes: int = 0
    rounds: int = 0
    inner_reps: int = 1

    def ratio(self, a: str, b: str) -> float:
        """``a`` ÷ ``b`` — >1 means ``a`` is slower."""
        return self.medians[a] / self.medians[b]

    def to_dict(self) -> dict:
        return {
            "evict_passes": self.evict_passes,
            "rounds": self.rounds,
            "inner_reps": self.inner_reps,
            "medians_us": {
                k: round(v, 2) for k, v in self.medians.items()
            },
            "stdevs_us": {
                k: round(v, 2) for k, v in self.stdevs.items()
            },
        }


def blocked_paired(
    fns: dict[str, Callable[[], object]],
    *,
    evict_passes: int = DEPLOY_EVICT_PASSES,
    rounds: int = 20,
    warm_per_block: int = 2,
    inner_reps: int = 4,
    flush: Callable[[int], None] | None = None,
    seed: int = 0,
) -> RegimeResult:
    """Time every callable under identical eviction, order shuffled.

    Each round visits all callables in a fresh shuffled order. Within a
    visit the callable runs `warm_per_block` times untimed (its weights
    come back into cache, as they would for a caller that just used it),
    then `inner_reps` calls are timed as one block. The median over
    rounds is the per-call time.

    Shuffling cancels clock drift across candidates; the untimed warm
    calls keep each measurement in the regime a real caller sees, rather
    than the evicted-every-call regime no caller ever sees.
    """
    if not fns:
        raise ValueError("no callables to measure")
    flush = flush or evictor()
    names = list(fns)
    samples: dict[str, list[float]] = {n: [] for n in names}
    gen = torch.Generator().manual_seed(seed)

    for fn in fns.values():  # compile / warm once outside the timing
        fn()
    torch.cuda.synchronize()

    for _ in range(max(1, rounds)):
        for i in torch.randperm(len(names), generator=gen).tolist():
            name = names[i]
            fn = fns[name]
            if evict_passes:
                flush(evict_passes)
            for _ in range(max(0, warm_per_block)):
                fn()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(max(1, inner_reps)):
                fn()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1e6
            samples[name].append(elapsed / max(1, inner_reps))

    return RegimeResult(
        medians={n: stats.median(v) for n, v in samples.items()},
        stdevs={
            n: (stats.stdev(v) if len(v) > 1 else 0.0)
            for n, v in samples.items()
        },
        samples=samples,
        evict_passes=evict_passes,
        rounds=rounds,
        inner_reps=inner_reps,
    )


def regime_curve(
    fns: dict[str, Callable[[], object]],
    levels: Sequence[int] = REGIME_LEVELS,
    **kwargs,
) -> dict[int, RegimeResult]:
    """Run :func:`blocked_paired` at each eviction level."""
    return {
        lvl: blocked_paired(fns, evict_passes=lvl, **kwargs)
        for lvl in levels
    }
