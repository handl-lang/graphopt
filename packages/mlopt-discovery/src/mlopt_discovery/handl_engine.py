"""``engine="handl"`` — the first-class differential backend.

Plan 0033 phase 1 (``project/plans/pl-0029-plan-0033-…``): the
``mlopt-handl-peer`` crate stops being a spike fixture and becomes a
saturation *checker*.  :class:`HandlEngine` is an :class:`EGraph`
subclass — so it satisfies the whole ``mlopt_core.ports.Engine``
port and ``mlopt_orchestrator.optimize.search(..., engine=...)``
accepts it directly — that runs the ordinary Python saturation and
then **replays every recorded rule application through handl's
``RewriteEngine``**, comparing the rewritten term structurally.

The honesty contract, per the plan:

* the Python ``EGraph`` remains the differential *oracle* — it does
  the saturating; handl checks each firing a second time;
* a structural disagreement **refuses loudly** —
  :class:`HandlDisagreementError` is raised before ``run`` returns,
  so a divergent engine can never ship a silent result;
* everything handl cannot express is *counted, not hidden* — laws
  whose rhs carries a leaf (the ``mlopt:const``/``mlopt:var`` bridge
  kinds are not comparable yet) or whose ``derive`` mints bindings
  the rule form cannot see, applications whose bindings do not
  resolve to terms, and laws the peer declines are reported in
  ``stats["handl_parity"]`` beside the usual funnel numbers.

``handl-egraph`` saturation proper is a later phase (the plan's
phase 4 perf gate); until that crate exists this engine is the
differential seat the plan asks for — every equivalence the Python
engine claims is checked twice wherever the peer is built.

Peer surfaces (same resolution order as ``tests/test_handl_peer.py``):

1. the PyO3 extension ``mlopt_handl_peer`` (``maturin develop`` in
   ``packages/mlopt-handl-peer``);
2. the CLI fallback ``src/bin/rewrite.rs`` — honoured via
   ``MLOPT_HANDL_PEER_BIN`` or ``~/GitHub/handl-lang/host/target/
   debug/rewrite``.

Neither present → :class:`HandlPeerUnavailable` at construction —
the honest-skip pattern: decline with the reason and the build
command, never silently run unchecked.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from mlopt_core.egraph import EGraph
from mlopt_core.egraph.terms import _term_instantiate
from mlopt_core.ir import Op

from mlopt_discovery import vocabdir

__all__ = [
    "HandlDisagreementError",
    "HandlEngine",
    "HandlPeer",
    "HandlPeerUnavailable",
    "is_structural_rhs",
    "render_term",
    "resolve_engine",
    "term_to_ast",
]


class HandlPeerUnavailable(RuntimeError):
    """Neither peer surface (extension nor CLI) is available."""


class HandlDisagreementError(RuntimeError):
    """handl's rewrite of an application disagrees with mlopt's."""


#: The fragment directory the peer loads — the canonical ``vocab/``
#: projection (plan 0033 phase 0), overridable per peer.
_DEFAULT_BIN = (
    Path.home() / "GitHub/handl-lang/host/target/debug/rewrite"
)


def _peer_bin() -> Path:
    return Path(os.environ.get("MLOPT_HANDL_PEER_BIN", _DEFAULT_BIN))


try:  # the shipping surface — absent until maturin builds it
    # importlib keeps ty from resolving the optional cdylib either way.
    _ext: Any = importlib.import_module("mlopt_handl_peer")
except ImportError:  # pragma: no cover - depends on the local build
    _ext = None


class HandlPeer:
    """Client for the ``mlopt-handl-peer`` crate — ext or CLI.

    ``fragments`` is the handl ``Language`` directory the peer loads;
    ``None`` resolves :func:`vocabdir.default_vocab_dir` (the repo
    ``vocab/`` projection — the same directory ``pipeline.propose``
    consumes, per the phase-0 exit gate).  Construction raises
    :class:`HandlPeerUnavailable` when neither surface exists or no
    fragment directory resolves — a differential engine with no
    differential partner is not a differential engine.
    """

    def __init__(self, fragments: str | Path | None = None) -> None:
        """Bind the fragment directory; refuse when no surface exists."""
        directory = (
            Path(fragments)
            if fragments is not None
            else vocabdir.default_vocab_dir()
        )
        if directory is None or not Path(directory).is_dir():
            raise HandlPeerUnavailable(
                "no handl fragment directory — pass ``fragments=`` or "
                "set MLOPT_VOCAB_DIR (vocabdir.default_vocab_dir() "
                "found none)"
            )
        self.fragments = Path(directory)
        self._ext: Any = _ext  # extension module — untyped cdylib
        self._bin = _peer_bin()
        if self._ext is None and not self._bin.is_file():
            raise HandlPeerUnavailable(
                "handl peer not built — neither the mlopt_handl_peer "
                "extension nor the rewrite CLI is available "
                f"(looked for {_self_bin_hint()}).  Build it with:\n"
                "  cd packages/mlopt-handl-peer && "
                "VIRTUAL_ENV=../../.venv maturin develop "
                "--target-dir ~/GitHub/handl-lang/host/target\n"
                "(the crate's path deps need the sibling handl "
                "checkouts — see packages/mlopt-handl-peer/README.md)"
            )

    @property
    def surface(self) -> str:
        """Which peer surface this client drives."""
        return "extension" if self._ext is not None else "cli"

    def transform_ids(self) -> list[str]:
        """Return the fragment directory's transform ids (extension only).

        The CLI exposes only ``rewrite`` — asking it for the id list
        raises :class:`HandlPeerUnavailable` naming the surface gap.
        """
        if self._ext is None:
            raise HandlPeerUnavailable(
                "transform_ids needs the mlopt_handl_peer extension; "
                "the CLI fallback exposes rewrite only"
            )
        return list(
            self._ext.language_transform_ids(str(self.fragments))
        )

    def apply_rule(
        self, rule_id: str, nodes: list[dict], at: int
    ) -> dict | None:
        """Apply *rule_id* at node *at*; the parsed result, or ``None``.

        ``None`` is the honest "the engine declined" answer — a load
        failure, a rule that does not fire, or a refused native —
        never an exception, so the differential can count declines
        separately from disagreements.
        """
        ast = json.dumps(nodes)
        if self._ext is not None:
            try:
                return json.loads(
                    self._ext.apply_rule(
                        str(self.fragments), ast, int(at), rule_id
                    )
                )
            except Exception:
                # PyValueError covers load failures, unknown rule ids
                # and refused natives alike — all are declines.
                return None
        try:
            out = subprocess.run(
                [
                    str(self._bin),
                    str(self.fragments),
                    rule_id,
                    str(int(at)),
                    ast,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        try:
            return json.loads(out.stdout)
        except json.JSONDecodeError:
            return None


def _self_bin_hint() -> str:
    return str(_peer_bin())


# ---------------------------------------------------------------------------
#  Term <-> handl AST — the same flatten/render the audit driver proved
# ---------------------------------------------------------------------------


def _parts(node: Any) -> tuple | None:
    """``(head, args)`` for an ``Op`` pattern/term; ``None`` otherwise."""
    if isinstance(node, Op):
        return (node.op, tuple(node.args))
    return None


def term_to_ast(term: Any) -> tuple[list[dict], int]:
    """Flatten *term* post-order to the peer's node list + root index.

    Node payloads carry only ``children`` — attrs are not part of the
    comparison surface (the same scope ``bench/drivers/audit/
    handl_differential.py`` measured: 24 laws agree, the rest are
    classified skipped/declined/guard, never silently compared).
    """
    arena: list[dict] = []
    root = _flatten(term, arena)
    return arena, root


def _flatten(node: Any, arena: list[dict]) -> int:
    parts = _parts(node)
    if parts is None:
        arena.append(
            {
                "kind": str(getattr(node, "name", node)),
                "payload": {"children": []},
            }
        )
        return len(arena) - 1
    kids = [_flatten(a, arena) for a in parts[1]]
    arena.append({"kind": parts[0], "payload": {"children": kids}})
    return len(arena) - 1


def render_term(node: Any) -> dict:
    """Render a mlopt term as ``{kind, children}`` (handl's shape)."""
    parts = _parts(node)
    if parts is None:
        return {
            "kind": str(getattr(node, "name", node)),
            "children": [],
        }
    return {
        "kind": parts[0],
        "children": [render_term(a) for a in parts[1]],
    }


def ast_result_term(result: dict) -> dict:
    """Render a peer ``apply_rule`` result's root as ``{kind, children}``."""
    arena = result["arena"]
    index = int(result["root_index"])
    return _from_handl(arena, index)


def _from_handl(arena: list[dict], index: int) -> dict:
    node = arena[index]
    payload = node.get("payload") or {}
    return {
        "kind": node["kind"],
        "children": [
            _from_handl(arena, i) for i in payload.get("children", [])
        ],
    }


def is_structural_rhs(node: Any) -> bool:
    """Whether the pattern is ops/metavars only — no concrete leaves.

    A rhs carrying a ``Const``/``Var``/``Param`` renders through the
    bridge's ``mlopt:``-namespaced leaf kinds, which are not yet
    comparable against mlopt's term rendering (plan 0030 step 3) —
    those applications are counted ``skipped``, exactly as the audit
    driver classifies them.
    """
    if isinstance(node, str):
        return True
    parts = _parts(node)
    if parts is None:
        return False
    return all(is_structural_rhs(a) for a in parts[1])


# ---------------------------------------------------------------------------
#  The engine
# ---------------------------------------------------------------------------


class HandlEngine(EGraph):
    """An ``EGraph`` that checks every firing through handl.

    Pass an instance (or this class) as ``engine=`` to
    ``mlopt_orchestrator.optimize.search`` — or anywhere the
    ``mlopt_core.ports.Engine`` surface is consumed::

        from mlopt_discovery.handl_engine import HandlEngine
        result = optimize.search(model, x, source=..., engine=HandlEngine())

    ``peer`` injects a :class:`HandlPeer` (or any object with the
    same ``apply_rule(rule_id, nodes, at)`` surface — tests use a
    fake); ``None`` builds one against ``fragments`` (default: the
    repo ``vocab/``).  Construction raises
    :class:`HandlPeerUnavailable` when no peer surface exists.

    After each :meth:`run` — and on demand via :meth:`check_pending`
    — every application recorded since the last check is replayed
    through ``transform.mlopt.<rule>`` at the matched term's root and
    the rendered result compared with the instantiated rhs.  The
    tally lands in ``stats["handl_parity"]``; a disagreement raises
    :class:`HandlDisagreementError`.
    """

    engine_name = "handl"

    def __init__(
        self,
        *args: Any,
        peer: Any = None,
        fragments: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the e-graph, then bind the differential peer."""
        super().__init__(*args, **kwargs)
        self._peer = peer if peer is not None else HandlPeer(fragments)
        self._handl_cursor = 0
        #: ``rule.name|rendered-lhs`` → peer outcome cache: identical
        #: firings cannot disagree with themselves, and saturation
        #: records thousands of them.
        self._handl_memo: dict[tuple, str] = {}
        self.handl_parity: dict[str, Any] = {
            "agree": 0,
            "declined": 0,
            "skipped": 0,
            "unresolved": 0,
            "disagreed": [],
            "peer_calls": 0,
        }

    # -- the Engine port -------------------------------------------------

    def run(
        self,
        rules: Any,
        root_eid: int,
        max_iterations: int = 100,
        max_nodes: int = 100_000,
        rule_budgets: dict[str, int] | None = None,
        stop: str = "fixed_point",
        patience: int = 3,
        cost_fn: Any = None,
        policy: Any = None,
    ) -> dict[str, Any]:
        """Saturate, then check every new application through handl."""
        stats = super().run(
            rules,
            root_eid,
            max_iterations=max_iterations,
            max_nodes=max_nodes,
            rule_budgets=rule_budgets,
            stop=stop,
            patience=patience,
            cost_fn=cost_fn,
            policy=policy,
        )
        self.check_pending()
        stats["handl_parity"] = self.parity_stats()
        return stats

    # -- the differential -------------------------------------------------

    def parity_stats(self) -> dict[str, Any]:
        """Return the handl-side tally — safe to embed in a result document."""
        return {
            **{
                k: v
                for k, v in self.handl_parity.items()
                if k != "disagreed"
            },
            "disagreed": list(self.handl_parity["disagreed"]),
            "surface": getattr(self._peer, "surface", "injected"),
            # ``_applications`` only exist at truncation_level >= 2 —
            # a level-1 engine checks nothing and says so.
            "tracked": bool(self._track),
        }

    def check_pending(self) -> None:
        """Replay every application recorded since the last check.

        Applications are the proof-level record ``EGraph.apply_rule``
        keeps under ``_track`` (the default truncation level): each
        entry carries the rule name and the fired substitution, which
        resolves to the concrete matched/rhs terms through
        ``_resolve_subst`` + ``_term_instantiate`` — the same replay
        machinery certificates use.
        """
        apps = self._applications
        for app in apps[self._handl_cursor :]:
            self._check_application(app)
        self._handl_cursor = len(apps)
        if self.handl_parity["disagreed"]:
            names = ", ".join(
                d["rule"] for d in self.handl_parity["disagreed"][:10]
            )
            raise HandlDisagreementError(
                "handl and mlopt disagree on "
                f"{len(self.handl_parity['disagreed'])} application(s): "
                f"{names} — the differential oracle refuses to ship "
                "an unchecked equivalence"
            )

    def _check_application(self, app: dict) -> None:
        parity = self.handl_parity
        rule = self._rule_objs.get(app.get("rule"))
        if rule is None:  # union recorded without a Rewrite object
            parity["unresolved"] += 1
            return
        # ``derive`` mints bindings (``$leaf:`` channels, extra
        # metavars) the handl rule form cannot see — the rewrite it
        # produces cannot be compared, same honesty class as a
        # leaf-carrying rhs.  ``check``/``cond``/``dspec`` are pure
        # vetoes: mlopt already evaluated them before recording the
        # application, and they do not change the rhs, so those laws
        # stay checked.
        if rule.derive is not None or not is_structural_rhs(rule.rhs):
            parity["skipped"] += 1
            return
        try:
            bound = self._resolve_subst(app.get("subst") or {})
            if bound is None:
                parity["unresolved"] += 1
                return
            lhs = _term_instantiate(rule.lhs, bound)
            rhs = _term_instantiate(rule.rhs, bound)
        except (KeyError, RecursionError):
            parity["unresolved"] += 1
            return
        nodes, root = term_to_ast(lhs)
        key = (rule.name, json.dumps(nodes, sort_keys=True))
        if key in self._handl_memo:
            outcome = self._handl_memo[key]
            # A repeated disagreement is already on the detail list —
            # ``disagreed`` counts *distinct* firings, not occurrences.
            if outcome != "disagree":
                parity[outcome] += 1
            return
        result = self._peer.apply_rule(
            f"transform.mlopt.{rule.name}", nodes, root
        )
        parity["peer_calls"] += 1
        if result is None:
            self._handl_memo[key] = "declined"
            parity["declined"] += 1
            return
        actual = ast_result_term(result)
        expected = render_term(rhs)
        if actual == expected:
            self._handl_memo[key] = "agree"
            parity["agree"] += 1
        else:
            self._handl_memo[key] = "disagree"
            parity["disagreed"].append(
                {
                    "rule": rule.name,
                    "lhs": render_term(lhs),
                    "mlopt_rhs": expected,
                    "handl_rhs": actual,
                }
            )


def resolve_engine(spec: Any) -> Any:
    """Resolve an ``engine=`` spec — ``"handl"`` included.

    ``mlopt_orchestrator.optimize.search`` takes engine *objects*;
    this is the graphopt-side spelling for the string form the plan
    names::

        search(..., engine=resolve_engine("handl"))

    ``None``/``"python"`` → the reference ``EGraph``; ``"handl"`` →
    a :class:`HandlEngine` (raising :class:`HandlPeerUnavailable`
    when the peer is not built); anything else passes through — an
    engine instance, class, or factory is returned unchanged for the
    caller's own ``engine=``.
    """
    if spec is None or spec == "python":
        return EGraph()
    if spec == "handl":
        return HandlEngine()
    return spec
