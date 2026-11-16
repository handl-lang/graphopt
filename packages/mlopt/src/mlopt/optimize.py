"""Optimize orchestration — lift → search → lower(verify) → report.

Composition only: every step delegates to ``mlopt_orchestrator`` /
``kit_torch``.  The report is the product contract — measured
before/after, verify status, and honest declines.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any

_EQSAT_TIMEOUT_ENV = "MLOPT_EQSAT_TIMEOUT"


def _eqsat_timeout_seconds() -> float | None:
    """``$MLOPT_EQSAT_TIMEOUT`` in seconds, or ``None`` when unset.

    Overrides the ``EqsatEngine`` 300 s default for models whose
    eqsat-run search legitimately runs longer.  ``None`` keeps the
    default engine path untouched; an unparsable or non-positive
    value is a named ``ValueError``, never a silent default.
    """
    raw = os.environ.get(_EQSAT_TIMEOUT_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{_EQSAT_TIMEOUT_ENV} must be a float (seconds), got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{_EQSAT_TIMEOUT_ENV} must be positive, got {raw!r}")
    return value


@dataclass
class OptimizeReport:
    model_ref: str
    backend: str
    verified: bool | None = None
    search_s: float = 0.0
    lower_s: float = 0.0
    n_enodes: int | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None
    #: The ``SearchResult`` handle when search succeeded — carries the
    #: extracted ``term`` and ``param_values`` ``mlopt export`` needs.
    #: ``LowerResult`` itself exposes only the delivered ``module``
    #: (an executor), not a term — this is the honest term source.
    search_result: Any = None
    #: ``--deliver`` measurements — median ms/call, min in parens on
    #: render.  ``None`` when no comparison ran (deliver off, or the
    #: pipeline failed before lowering).
    eager_ms: float | None = None
    delivered_ms: float | None = None
    eager_min_ms: float | None = None
    delivered_min_ms: float | None = None
    speedup: float | None = None
    runner_name: str | None = None
    #: ``"game"`` / ``"heuristic"`` — which mechanism picked the
    #: delivery runner (``None`` when ``--deliver`` never ran).  The
    #: honesty contract: the report always names its decider.
    deliver_mechanism: str | None = None
    #: Verify-decline detail — populated only when ``lower(verify=True)``
    #: raised ``VerifyDeclinedError``: the declined report's verdict
    #: numbers and which contract decided (``stats["accepted_by"]``).
    #: ``None`` on every other path — a declined gate is a *failed*
    #: gate, so ``verified`` is ``False`` there, never ``None``.
    verify_max_abs: float | None = None
    verify_max_rel: float | None = None
    verify_accepted_by: str | None = None

    def render(self) -> str:
        decline = ""
        if self.verify_max_abs is not None:
            decline = f"  declined max_abs={self.verify_max_abs:.3e}"
            if self.verify_accepted_by:
                decline += f" accepted_by={self.verify_accepted_by}"
        lines = [
            f"model:    {self.model_ref}",
            f"backend:  {self.backend}",
            f"search:   {self.search_s:.1f}s"
            + (f"  enodes={self.n_enodes}" if self.n_enodes is not None else ""),
            f"lower:    {self.lower_s:.1f}s  verify={self.verified}{decline}",
        ]
        lines.extend(f"note:     {n}" for n in self.notes)
        if self.runner_name is not None or self.eager_ms is not None:
            eager = f"{self.eager_ms:.2f}ms" if self.eager_ms is not None else "n/a"
            if self.eager_min_ms is not None:
                eager += f" (min {self.eager_min_ms:.2f})"
            delivered = (
                f"{self.delivered_ms:.2f}ms" if self.delivered_ms is not None else "eager only"
            )
            if self.delivered_min_ms is not None:
                delivered += f" (min {self.delivered_min_ms:.2f})"
            speedup = f"  speedup={self.speedup:.2f}x" if self.speedup is not None else ""
            mech = f"  mechanism={self.deliver_mechanism}" if self.deliver_mechanism else ""
            lines.append(
                f"deliver:  runner={self.runner_name or 'none'}  "
                f"eager={eager}  delivered={delivered}{speedup}{mech}"
            )
        if self.error:
            lines.append(f"error:    {self.error}")
        return "\n".join(lines)


def optimize_model(
    spec: Any,
    *,
    backend: str = "torch",
    rules: str | None = None,
    max_iterations: int = 4,
    verify: bool = True,
    runner: Any = None,
    deliver: bool = False,
    bench_iters: int = 50,
    deliver_budget: int | None = None,
    deliver_player: str = "playbook",
    deliver_heuristic: bool = False,
) -> tuple[Any, OptimizeReport]:
    """Run the pipeline over an adapter's ``ModelSpec``.

    Returns ``(lowered, report)``.  ``lowered`` is whatever
    ``Optimizer.lower`` returns (a ``LowerResult``); on failure the
    report carries ``error`` and ``lowered`` is None.

    ``deliver=True`` runs the fast-artifact pass
    (:mod:`mlopt.deliver`): the eager lowering is verified first,
    then re-lowered under a runner — verify still gates the
    DELIVERED module — and eager-vs-delivered is timed over
    ``bench_iters`` forwards.  ``runner=`` supplies the runner
    explicitly and skips selection.

    Runner *selection* has two mechanisms, and the report names
    which decided (``deliver_mechanism``):

    * ``deliver_budget`` set (and not ``deliver_heuristic``) — the
      **game** (:mod:`mlopt.deliver_game`): a gamekit episode where
      the named ``deliver_player`` spends up to ``deliver_budget``
      moves evaluating runner configurations; the commit is the
      fastest *measured, verify-gated* config — eager included.
      Falls back to the heuristic with a note when gamekit is
      absent or the player is unseated.
    * otherwise — the **heuristic** (:func:`mlopt.deliver.
      select_runner`): the deterministic region_fuse/cuda_graph
      pick, unchanged.

    ``backend`` must be ``"torch"`` — the only backend
    ``Optimizer`` is wired for today; anything else raises
    ``ValueError`` rather than recording a backend the pipeline
    never used.
    """
    if backend != "torch":
        raise ValueError(
            f"backend={backend!r} is not wired — optimize_model always "
            "runs Optimizer(backend=TorchBackend()); only 'torch' is "
            "honest today (the CLI's --backend choices agree)"
        )
    from kit_torch.backend import TorchBackend
    from mlopt_orchestrator import VerifyDeclinedError  # lazy: torch-free import path
    from mlopt_orchestrator.optimize import Optimizer

    report = OptimizeReport(model_ref=getattr(spec, "meta", {}).get("ref", "?"), backend=backend)
    # Leaf-provenance honesty (the .hnl source): shapes-not-data
    # artifacts get seeded synthetic values unless feed=<json> supplied
    # real ones — the verify gate is real either way, but the report
    # must not let synthetic weights pass as the artifact's own.
    leaf_src = (getattr(spec, "meta", None) or {}).get("leaf_source")
    if leaf_src in ("synthetic", "mixed"):
        report.notes.append(
            f"leaf values: {leaf_src} — .hnl artifacts carry shapes, "
            "not data; pass --adapter-opt feed=<json> for real values"
        )
    # Input-probe honesty (the onnx-rust/arch-rust sources): the spec
    # declares shapes, not data — verify runs on seeded synthetic
    # inputs unless the adapter was fed real ones.  ``leaf_source``
    # stays "real" (the weights ARE the artifact's own), so this needs
    # its own channel.
    input_src = (getattr(spec, "meta", None) or {}).get("input_source")
    if input_src in ("synthetic", "mixed"):
        report.notes.append(
            f"input values: {input_src} — the spec declares shapes, "
            "not data; pass --adapter-opt inputs='{\"name\": [dims]}' or "
            "feed=<json> for real values"
        )

    # A spec may carry its own graph-source port (``.hnl`` sources
    # lift text → term at load; ``source.to_ir`` returns the
    # pre-lifted IR).  ``None`` keeps the backend's TorchSource.
    # ``$MLOPT_EQSAT_TIMEOUT`` raises the eqsat engine's 300 s search
    # bound — passed as a zero-arg factory so the eqsat-run binary
    # still resolves lazily at search time, same as the default path.
    eqsat_timeout = _eqsat_timeout_seconds()
    engine = None
    if eqsat_timeout is not None:
        from mlopt_orchestrator.eqsat import EqsatEngine

        engine = partial(EqsatEngine, timeout=eqsat_timeout)
        report.notes.append(f"eqsat search timeout: {eqsat_timeout:g}s via {_EQSAT_TIMEOUT_ENV}")
    optz = Optimizer(
        backend=TorchBackend(),
        source=getattr(spec, "source", None),
        rules=rules,
        engine=engine,
    )
    inputs = spec.inputs

    t0 = time.time()
    try:
        res = optz.search(spec.module, inputs, max_iterations=max_iterations)
    except Exception as e:  # noqa: BLE001 — report honestly, don't hide
        report.error = f"search failed: {type(e).__name__}: {e}"
        return None, report
    report.search_s = time.time() - t0
    report.n_enodes = getattr(res, "n_enodes", None)
    report.search_result = res

    t0 = time.time()
    try:
        # deliver=True always lowers eager first — the delivered
        # runner (selected or caller-supplied) is applied by the
        # deliver pass so verify gates the DELIVERED module and the
        # eager timing baseline exists.
        lowered = optz.lower(res, inputs, verify=verify, runner=None if deliver else runner)
    except VerifyDeclinedError as e:
        # The verify gate refused — a FAILED gate, not "no gate":
        # ``verified`` is False, and the declined LowerResult's
        # verdict (``.verified`` report + ``stats["accepted_by"]``)
        # rides the report instead of being flattened away.
        report.lower_s = time.time() - t0
        declined = getattr(e, "result", None)
        gate = getattr(declined, "verified", None) if declined is not None else None
        report.verified = bool(getattr(gate, "passed", False)) if gate is not None else False
        if gate is not None:
            report.verify_max_abs = getattr(gate, "max_abs", None)
            report.verify_max_rel = getattr(gate, "max_rel", None)
        stats = getattr(declined, "stats", None)
        if isinstance(stats, dict):
            accepted = stats.get("accepted_by")
            if isinstance(accepted, str):
                report.verify_accepted_by = accepted
        report.error = f"lower/verify failed: {type(e).__name__}: {e}"
        return None, report
    except Exception as e:  # noqa: BLE001
        report.lower_s = time.time() - t0
        report.error = f"lower/verify failed: {type(e).__name__}: {e}"
        return None, report
    report.lower_s = time.time() - t0
    gate = getattr(lowered, "verified", None)
    if gate is None:
        report.verified = None
    elif hasattr(gate, "passed"):
        report.verified = bool(gate.passed)
    else:
        report.verified = bool(gate)

    if deliver:
        from mlopt.deliver import Delivery
        from mlopt.deliver import deliver as _deliver

        outcome: Delivery | None = None
        t0 = time.time()
        # A non-positive --deliver-budget used to fall back to the
        # heuristic silently — name it in the report instead.
        if deliver_budget is not None and int(deliver_budget) <= 0:
            report.notes.append(
                f"deliver: --deliver-budget {int(deliver_budget)} <= 0 — "
                "the delivery game needs a positive move budget; "
                "the game path was skipped"
            )
        call_kwargs = dict(getattr(spec, "kwargs", None) or {})
        use_game = (
            deliver_budget is not None
            and int(deliver_budget) > 0
            and not deliver_heuristic
            and runner is None
        )
        if use_game:
            try:
                from mlopt.deliver_game import GameUnavailable, deliver_game

                outcome = deliver_game(
                    optz=optz,
                    res=res,
                    lowered_eager=lowered,
                    inputs=inputs,
                    meta=getattr(spec, "meta", None) or {},
                    bench_iters=bench_iters,
                    verify=verify,
                    budget=int(deliver_budget),
                    player=deliver_player,
                    call_kwargs=call_kwargs,
                )
            except GameUnavailable as e:
                report.notes.append(
                    f"deliver: game unavailable ({e}) — fell back to the heuristic runner selection"
                )
                outcome = None
            except Exception as e:  # noqa: BLE001 — game must not sink the run
                report.notes.append(
                    f"deliver: game path failed ({type(e).__name__}: {e}) — "
                    "fell back to the heuristic runner selection"
                )
                outcome = None
        if outcome is None:
            try:
                outcome = _deliver(
                    optz=optz,
                    res=res,
                    lowered_eager=lowered,
                    inputs=inputs,
                    meta=getattr(spec, "meta", None) or {},
                    bench_iters=bench_iters,
                    verify=verify,
                    runner=runner,
                    call_kwargs=call_kwargs,
                )
            except Exception as e:  # noqa: BLE001 — deliver must not sink the run
                report.lower_s += time.time() - t0
                report.notes.append(
                    f"deliver: pass failed ({type(e).__name__}: {e}) — "
                    "shipped the verified eager lowering"
                )
                return lowered, report
        report.lower_s += time.time() - t0
        lowered = outcome.lowered
        report.runner_name = outcome.runner_name
        report.deliver_mechanism = getattr(outcome, "mechanism", "heuristic")
        report.notes.extend(outcome.notes)
        if outcome.eager_med_ms is not None:
            report.eager_ms = outcome.eager_med_ms
            report.eager_min_ms = outcome.eager_min_ms
        if outcome.delivered_med_ms is not None:
            report.delivered_ms = outcome.delivered_med_ms
            report.delivered_min_ms = outcome.delivered_min_ms
            if outcome.delivered_med_ms > 0 and outcome.eager_med_ms:
                report.speedup = outcome.eager_med_ms / outcome.delivered_med_ms
        # The shipped module may differ from the first lowering —
        # re-derive the verify flag from the delivered result.
        gate = getattr(lowered, "verified", None)
        if gate is None:
            report.verified = None
        elif hasattr(gate, "passed"):
            report.verified = bool(gate.passed)
        else:
            report.verified = bool(gate)
    return lowered, report
