"""Task-aware certificates — the ``task=`` / ``task_tol=`` seam (plan 0015).

Pins the task-metric gate end to end:

* ``mlopt_core.metrics`` — the shipped :class:`TaskMetric`
  implementations (``MaxRel`` / ``TopKAgreement`` / ``ArgmaxStability``
  / ``LogitKL`` / ``CosineRanking``) on plain lists, torch tensors and
  numpy arrays; the output-pytree pairing (``tuple``/``dict``) and the
  loud structural errors.
* ``search(..., task=..., task_tol=...)`` records the contract —
  ``stats["task"]`` plus ``result.task`` / ``result.task_tol`` — and
  ``lower`` evaluates it on the verify input: the task distance IS the
  verdict, with the pointwise measurements still reported.
* The bound_amplification finding made flesh: a certified bounded
  member whose measured drift exceeds a tight pointwise ``atol`` is
  declined by the pointwise gate (``accepted_by="declined"``) but the
  top-1 task metric stays perfect — under ``task=ArgmaxStability()``
  the same delivery passes with ``accepted_by="task"`` while the
  bound ledger keeps its measured values.  And a
  distribution-shifted input flips the argmax — the task gate
  declines (the certificate is conditioned on the calibration input).
* ``export_optimized(..., stats=...)`` — the manifest carries the
  task fields verbatim.
"""

from unittest import mock

import pytest
import torch
import torch.nn as nn
from mlopt_core import metrics
from mlopt_core.cost import flops_cost
from mlopt_core.ir import op_repr
from mlopt_core.metrics import (
    ArgmaxStability,
    CosineRanking,
    LogitKL,
    MaxRel,
    TopKAgreement,
    _RowMetric,
)
from mlopt_core.ports import TaskMetric
from mlopt_orchestrator import optimize as opt_mod
from mlopt_orchestrator.optimize import (
    Compositional,
    Optimizer,
    _task_contract,
)
from mlopt_torch.backend import TorchBackend
from mlopt_torch.export import export_optimized


def _opt():
    return Optimizer(backend=TorchBackend())


# ---------------------------------------------------------------------------
#  Model fixtures
# ---------------------------------------------------------------------------


class _NearDead(nn.Module):
    """One Linear whose middle output rows are nearly dead.

    Under ``detect_specials`` + ``error_budget`` the bounded
    ``elide_bounded`` member zeroes the near-dead rows — real drift
    (~1e-3 rel) that honors the propagated bound, exceeds any tight
    ``atol``, and never moves the argmax (dead rows produce ~0 in
    either form).
    """

    def __init__(self, i=32, o=64, eps=8e-4, scale=5.0):
        super().__init__()
        self.lin = nn.Linear(i, o, bias=False).double()
        with torch.no_grad():
            self.lin.weight.mul_(scale)
            self.lin.weight[16:48] = 0.0
            idx = torch.arange(16, 48)
            self.lin.weight[idx, idx % i] = eps

    def forward(self, t):
        return self.lin(t)


class _Shifted(nn.Module):
    """Near-dead rows respond only to input column 3.

    Live rows carry a small *positive* col-3 entry (below the dead
    rows' eps): an input concentrated on column 3 makes the reference
    argmax land on a dead row (eps·K) while the bounded member's
    zeroed rows sit at 0 below the live rows' positive outputs — the
    argmax flips.  On generic inputs the argmax is stable.
    """

    def __init__(self, i=32, o=64, eps=8e-4):
        super().__init__()
        self.lin = nn.Linear(i, o, bias=False).double()
        with torch.no_grad():
            self.lin.weight.mul_(5.0)
            self.lin.weight[16:48] = 0.0
            self.lin.weight[16:48, 3] = eps
            self.lin.weight[:16, 3] = 1e-4
            self.lin.weight[48:, 3] = 1e-4

    def forward(self, t):
        return self.lin(t)


class _LowRank(nn.Module):
    """A dense Linear whose stored weight is exactly rank-4 — the
    ``detect_factors`` pass offers the certified two-GEMM member,
    whose reassociated evaluation drifts by ~1e-15 (pointwise-failing
    under a tight rtol, task-perfect)."""

    def __init__(self, i=64, o=32, r=4):
        super().__init__()
        self.lin = nn.Linear(i, o, bias=False).double()
        with torch.no_grad():
            g = torch.Generator().manual_seed(5)
            a = torch.randn(o, r, generator=g, dtype=torch.float64)
            b = torch.randn(r, i, generator=g, dtype=torch.float64)
            self.lin.weight.copy_(a @ b)

    def forward(self, t):
        return self.lin(t)


class _BoundBlock(nn.Module):
    """One near-dead Linear — the compositional bounded fixture."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(32, 32, bias=False).double()
        with torch.no_grad():
            self.lin.weight.mul_(5.0)
            self.lin.weight[8:24] = 0.0
            idx = torch.arange(8, 24)
            self.lin.weight[idx, idx % 32] = 8e-4

    def forward(self, t):
        return self.lin(t)


class _BoundStack(nn.Module):
    """Two structurally-identical bounded blocks (the second replays
    through the compositional cache)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*[_BoundBlock() for _ in range(2)])

    def forward(self, t):
        return self.net(t)


# ---------------------------------------------------------------------------
#  Metric units — plumbing (_rows / _leaves / _match / _RowMetric)
# ---------------------------------------------------------------------------


















def test_metrics_accept_torch_and_numpy():
    np = pytest.importorskip("numpy")
    ref_t = torch.tensor([[1.0, 3.0, 2.0], [4.0, 1.0, 0.0]])
    opt_t = torch.tensor([[1.1, 3.0, 2.0], [0.0, 1.0, 4.0]])
    for m in (MaxRel(), TopKAgreement(1), ArgmaxStability(), LogitKL()):
        d_t = m.distance(ref_t, opt_t)
        d_np = m.distance(ref_t.numpy(), opt_t.numpy())
        d_l = m.distance(ref_t.tolist(), opt_t.tolist())
        assert d_t == pytest.approx(d_np)
        assert d_t == pytest.approx(d_l)
    emb_t = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert CosineRanking().distance(emb_t, emb_t.numpy()) == 0.0
    assert isinstance(MaxRel().distance(np.eye(3), np.eye(3)), float)






def test_internal_helpers():
    assert metrics._argmax([1.0, 3.0, 2.0]) == 1
    assert metrics._topk_idx([3.0, 1.0, 2.0], 2) == {0, 2}
    # list-of-tensor-likes pool into one row set.
    rows = metrics._rows([torch.ones(2, 2), torch.zeros(1, 2)])
    assert rows == [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]

    class _NoCallTolist:
        tolist = "not-callable"

    with pytest.raises(TypeError):
        metrics._rows(_NoCallTolist())


# ---------------------------------------------------------------------------
#  The contract — _task_contract validation
# ---------------------------------------------------------------------------




def test_run_exec_dispatch():
    """``_run_exec`` prefers ``forward``; a bare callable still runs;
    non-cloneable args pass through."""
    mod = mock.Mock()
    mod.forward = lambda *a: ("fwd", *a)
    assert opt_mod._run_exec(mod, (1, 2)) == ("fwd", 1, 2)

    def bare(*a):
        return a

    assert opt_mod._run_exec(bare, (3.5,)) == (3.5,)
    t = torch.ones(2)
    out = opt_mod._run_exec(lambda x: x + 1, (t,))
    assert torch.equal(out, torch.full((2,), 2.0))


# ---------------------------------------------------------------------------
#  search / lower plumbing
# ---------------------------------------------------------------------------


def test_search_records_task_contract():
    torch.manual_seed(0)
    model = _NearDead().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    res = _opt().search(model, x, task=TopKAgreement(5), task_tol=0.1)
    assert res.task is not None and res.task.name == "top5_agreement"
    assert res.task_tol == 0.1
    # Declared at search; distance/passed land at verify time.
    assert res.stats["task"] == {
        "name": "top5_agreement",
        "tolerance": 0.1,
    }


def test_search_without_task_unchanged():
    torch.manual_seed(1)
    model = _NearDead().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    res = _opt().search(model, x)
    assert res.task is None and res.task_tol is None
    assert "task" not in res.stats
    low = _opt().lower(res, x)
    assert low.verified.passed
    assert low.stats["verify_metric"] == "max_rel"
    assert low.stats["accepted_by"] == "pointwise"


def test_task_tol_without_task_raises():
    torch.manual_seed(2)
    model = _NearDead().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    with pytest.raises(TypeError, match="requires a task="):
        _opt().search(model, x, task_tol=0.1)
    res = _opt().search(model, x)
    with pytest.raises(TypeError, match="requires a task="):
        _opt().lower(res, x, task_tol=0.1)


# ---------------------------------------------------------------------------
#  Evidence — task-accept where the pointwise gate declined
# ---------------------------------------------------------------------------


def test_task_accept_bounded_member_pointwise_atol_declined():
    """The bound_amplification finding, made flesh.

    The certified bounded member's measured drift (max_abs ~2.6e-3)
    is *honored by the propagated bound* but exceeds a tight
    ``atol=1e-6`` — the pointwise gate declines
    (``accepted_by="declined"``).  The top-1 task contract stays
    perfect, so the same delivery is accepted under
    ``task=ArgmaxStability()`` — ``accepted_by="task"``, and the
    pointwise bound AND the task distance are both on the record.
    """
    torch.manual_seed(7)
    model = _NearDead().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    opt = _opt()
    res = opt.search(
        model,
        x,
        detect_specials=True,
        error_budget=1e-3,
        cost_fn=flops_cost,
    )
    assert "__bl" in op_repr(res.term)

    # The pointwise gate declines: measured drift >> atol, though the
    # propagated bound is honored.
    declined = opt.lower(res, x, atol=1e-6)
    assert not declined.verified.passed
    assert declined.verified.max_abs > 1e-6
    assert declined.stats["verify_metric"] == "bound"
    assert declined.stats["accepted_by"] == "declined"
    assert declined.stats["error_bounds_honored"] is True

    # The task contract delivers it: distance 0, both numbers recorded.
    low = opt.lower(res, x, atol=1e-6, task=ArgmaxStability())
    assert low.verified.passed
    assert low.verified.max_rel == pytest.approx(
        declined.verified.max_rel
    )
    task = low.stats["task"]
    assert task == {
        "name": "argmax_stability",
        "tolerance": 0.0,
        "distance": 0.0,
        "passed": True,
        "evaluated_on": "verify_input",
    }
    assert low.stats["verify_metric"] == "argmax_stability"
    assert low.stats["accepted_by"] == "task"
    # The bound ledger is never silent — measured drift + accepted_by
    # ride each entry.
    e = low.stats["error_bounds"][0]
    assert e["accepted_by"] == "task"
    assert e["measured_max_rel"] == pytest.approx(low.verified.max_rel)
    assert e["output_bound"] > 0.0
    assert low.stats["error_bounds_honored"] is True


def test_task_accept_via_search_contract_and_optimize():
    """A contract declared at search gates the plain lower call; and
    ``optimize(..., task=..., task_tol=...)`` threads through the
    Monolithic partition."""
    torch.manual_seed(9)
    model = _NearDead().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    opt = _opt()
    res = opt.search(
        model,
        x,
        detect_specials=True,
        error_budget=1e-3,
        cost_fn=flops_cost,
        task=LogitKL(),
        task_tol=1e-6,
    )
    # No task kwarg needed at lower — the recorded contract governs.
    low = opt.lower(res, x, atol=1e-6)
    assert low.verified.passed
    assert low.stats["task"]["name"] == "logit_kl"
    assert low.stats["task"]["tolerance"] == 1e-6
    assert low.stats["accepted_by"] == "task"
    # An explicit task_tol= overrides the recorded tolerance; an
    # explicit task= overrides the metric (its own tolerance then
    # applies).
    low_tol = opt.lower(res, x, atol=1e-6, task_tol=1e-2)
    assert low_tol.stats["task"]["tolerance"] == 1e-2
    assert low_tol.verified.passed
    low2 = opt.lower(res, x, atol=1e-6, task=TopKAgreement(1))
    assert low2.stats["verify_metric"] == "top1_agreement"
    assert low2.verified.passed
    # A tuple-shaped verify input evaluates the metric on its args.
    low_tup = opt.lower(res, (x,), atol=1e-6)
    assert low_tup.verified.passed
    assert 0.0 < low_tup.stats["task"]["distance"] < 1e-6
    # The full pipeline spelling.
    mod, stats = opt.optimize(
        _NearDead().eval(),
        x,
        detect_specials=True,
        error_budget=1e-3,
        cost_fn=flops_cost,
        atol=1e-6,
        task=ArgmaxStability(),
    )
    assert stats["accepted_by"] == "task"
    assert mod is not None


def test_task_accept_exact_member_under_tight_rtol():
    """A certified *exact* factored member (bound 0 in practice,
    Frobenius ~1e-14) drifts ~1e-15 — failing a tight ``rtol`` while
    the task contract stays perfect."""
    torch.manual_seed(11)
    model = _LowRank().eval()
    x = torch.randn(8, 64, dtype=torch.float64)
    opt = _opt()
    res = opt.search(
        model, x, detect_factors=True, cost_fn=flops_cost
    )
    assert "__lr" in op_repr(res.term)
    declined = opt.lower(res, x, rtol=1e-18)
    assert declined.verified.max_rel > 1e-18
    assert not declined.verified.passed
    assert declined.stats["verify_metric"] == "max_rel"
    low = opt.lower(res, x, rtol=1e-18, task=ArgmaxStability())
    assert low.verified.passed
    assert low.stats["task"]["distance"] == 0.0
    assert low.stats["accepted_by"] == "task"


def test_task_decline_on_distribution_shift():
    """The certificate is calibration-conditioned: the same delivery
    verified on an input OUTSIDE the calibration distribution flips
    the argmax — the task gate declines (distance 1.0 > 0)."""
    torch.manual_seed(13)
    model = _Shifted().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    opt = _opt()
    res = opt.search(
        model,
        x,
        detect_specials=True,
        error_budget=1e-3,
        cost_fn=flops_cost,
        task=ArgmaxStability(),
    )
    # Calibrated input: the task contract holds.
    ok = opt.lower(res, x)
    assert ok.verified.passed
    assert ok.stats["task"]["distance"] == 0.0
    # Shifted input concentrated on the dead rows' trigger column:
    # the reference argmax lands on a row the member zeroed.
    shifted = torch.zeros(8, 32, dtype=torch.float64)
    shifted[:, 3] = 1e6
    bad = opt.lower(res, shifted)
    assert not bad.verified.passed
    assert bad.stats["task"]["distance"] == 1.0
    assert bad.stats["task"]["passed"] is False
    assert bad.stats["accepted_by"] == "declined"
    # The pointwise drift is still reported on the same record.
    assert bad.verified.max_abs > 0.0


def test_task_tol_override_changes_verdict():
    """``task_tol`` is the gate: a drift measured at 4e-4 passes at
    1e-3 and fails at 1e-9 under ``MaxRel``."""
    torch.manual_seed(15)
    model = _NearDead().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    opt = _opt()
    res = opt.search(
        model,
        x,
        detect_specials=True,
        error_budget=1e-3,
        cost_fn=flops_cost,
    )
    low = opt.lower(res, x, task=MaxRel(), task_tol=1e-3)
    assert low.verified.passed
    assert low.stats["task"]["name"] == "max_rel"
    assert low.stats["task"]["tolerance"] == 1e-3
    assert low.stats["verify_metric"] == "max_rel"
    tight = opt.lower(res, x, task=MaxRel(), task_tol=1e-9)
    assert not tight.verified.passed
    assert tight.stats["accepted_by"] == "declined"


def test_task_not_evaluated_without_verify():
    """``verify=False`` leaves the declared contract — name and
    tolerance — without a measured distance."""
    torch.manual_seed(17)
    model = _NearDead().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    opt = _opt()
    res = opt.search(model, x, task=TopKAgreement(1))
    low = opt.lower(res, x, verify=False)
    assert low.verified is None
    assert low.stats["task"] == {
        "name": "top1_agreement",
        "tolerance": 0.0,
    }
    assert "verify_metric" not in low.stats
    assert "accepted_by" not in low.stats




# ---------------------------------------------------------------------------
#  Compositional plumbing + manifest
# ---------------------------------------------------------------------------


def test_compositional_task_gate():
    """``optimize(strategy=Compositional(), task=...)`` — each block's
    verify gates on the task distance measured on the captured input;
    the record lands in the block report and its stats."""
    torch.manual_seed(19)
    model = _BoundStack().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    _mod, stats = _opt().optimize(
        model,
        x,
        strategy=Compositional(),
        error_budget=1e-3,
        detect_specials=True,
        cost_fn=flops_cost,
        task=ArgmaxStability(),
        verbose=False,
    )
    assert stats["n_optimized"] == 2
    for rep in stats["blocks"].values():
        assert rep["status"] == "optimized"
        t = rep["task"]
        assert t["name"] == "argmax_stability"
        assert t["distance"] == 0.0 and t["passed"] is True
        assert t["evaluated_on"] == "captured_input"
        assert rep["stats"]["accepted_by"] == "task"
        assert rep["stats"]["verify_metric"] == "argmax_stability"


def test_compositional_task_decline_keeps_original():
    """A task gate that fails declines the block — the drifted form is
    not grafted, and the failure is recorded."""
    torch.manual_seed(21)
    model = _BoundStack().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    _mod, stats = _opt().optimize(
        model,
        x,
        strategy=Compositional(),
        error_budget=1e-3,
        detect_specials=True,
        cost_fn=flops_cost,
        # The bounded member's real drift is ~1e-3 — MaxRel(0) fails.
        task=MaxRel(),
        task_tol=0.0,
        verbose=False,
    )
    assert stats["n_optimized"] == 0
    for rep in stats["blocks"].values():
        assert rep["status"] == "failed"
        assert "task" in rep.get("error", "")


class _PlainStack(nn.Module):
    """Two plain Linear blocks — structurally identical, so the
    second replays through the compositional search cache."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            *[nn.Linear(32, 32, bias=False).double() for _ in range(2)]
        )

    def forward(self, t):
        return self.net(t)


class _AlwaysFar:
    """A caller-defined duck metric — not a shipped class — that
    always reports maximal distance."""

    name = "always_far"
    tolerance = 0.0

    def distance(self, ref, opt):
        return 1.0


def test_compositional_task_cache_hit():
    """A task contract survives the structural-cache replay: the
    second (identical-structure) block verifies through the task gate
    on the cache-hit path."""
    torch.manual_seed(23)
    model = _PlainStack().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    _mod, stats = _opt().optimize(
        model,
        x,
        strategy=Compositional(),
        task=ArgmaxStability(),
        verbose=False,
    )
    assert stats["n_optimized"] == 2
    by_name = stats["blocks"]
    hit = next(r for r in by_name.values() if r.get("cache") == "hit")
    assert hit["task"]["name"] == "argmax_stability"
    assert hit["stats"]["accepted_by"] == "task"


def test_compositional_task_cache_hit_decline():
    """The cache-hit task gate fails closed too: a metric that always
    reports distance 1 declines the replay (falls back to search) and
    then the fresh block."""
    torch.manual_seed(25)
    model = _PlainStack().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    _mod, stats = _opt().optimize(
        model,
        x,
        strategy=Compositional(),
        task=_AlwaysFar(),
        verbose=False,
    )
    assert stats["n_optimized"] == 0
    assert all(
        r["status"] == "failed" for r in stats["blocks"].values()
    )


def test_export_manifest_carries_task_fields(tmp_path):
    """``task`` / ``verify_metric`` / ``accepted_by`` ride into the
    manifest verbatim — a task-gated delivery is never silent."""
    mod = nn.Linear(4, 4).eval()
    stats = {
        "task": {
            "name": "argmax_stability",
            "tolerance": 0.0,
            "distance": 0.0,
            "passed": True,
            "evaluated_on": "verify_input",
        },
        "verify_metric": "argmax_stability",
        "accepted_by": "task",
        "error_bounds": [
            {"rule": "weight_special#3", "bound": 8e-4,
             "accepted_by": "task"}
        ],
    }
    manifest = export_optimized(
        None,
        mod,
        tmp_path / "w.safetensors",
        fmt="safetensors",
        stats=stats,
    )
    assert manifest["task"] == stats["task"]
    assert manifest["verify_metric"] == "argmax_stability"
    assert manifest["accepted_by"] == "task"
    assert manifest["error_bounds"][0]["accepted_by"] == "task"
    # A stats dict without task keys adds nothing.
    m2 = export_optimized(
        None,
        mod,
        tmp_path / "w2.safetensors",
        fmt="safetensors",
        stats={"runner": "x"},
    )
    assert "task" not in m2 and "verify_metric" not in m2
    m3 = export_optimized(
        None, mod, tmp_path / "w3.safetensors", fmt="safetensors"
    )
    assert "accepted_by" not in m3
