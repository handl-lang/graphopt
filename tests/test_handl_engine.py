"""Plan 0033 phase 1 — ``engine="handl"``, the differential backend.

:class:`mlopt_discovery.handl_engine.HandlEngine` is an ``EGraph``
that replays every recorded rule application through the
``mlopt-handl-peer`` crate and refuses loudly on disagreement.  The
peer is a maturin/CLI artifact that is not part of the wheel, so the
unit tests inject a fake peer and the one end-to-end test skips
without a real build — the same honest-skip pattern
``test_handl_peer.py`` uses.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from mlopt_core.egraph import EGraph, Rewrite
from mlopt_core.ir import Const, Op, Var
from mlopt_core.laws import ALL_RULES
from mlopt_discovery import handl_engine
from mlopt_discovery.handl_engine import (
    HandlDisagreementError,
    HandlEngine,
    HandlPeer,
    HandlPeerUnavailable,
    is_structural_rhs,
    render_term,
    resolve_engine,
    term_to_ast,
)

_REPO = Path(__file__).resolve().parents[1]
_VOCAB = _REPO / "vocab"


# ---------------------------------------------------------------------------
#  Fakes — a peer surface with no Rust build
# ---------------------------------------------------------------------------


class _IdentityPeer:
    """Returns the input unchanged — a guaranteed *disagreement* for
    any non-identity law, and the declined path when told to."""

    surface = "fake"

    def __init__(self, decline: bool = False) -> None:
        self.calls: list[tuple] = []
        self._decline = decline

    def apply_rule(self, rule_id, nodes, at):
        self.calls.append((rule_id, nodes, at))
        if self._decline:
            return None
        return {
            "rewrites": 1,
            "root_index": at,
            "root": nodes[at],
            "arena": nodes,
        }


class _SwapPeer:
    """Swaps the root node's two children — comm_add's rewrite."""

    surface = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def apply_rule(self, rule_id, nodes, at):
        self.calls.append((rule_id, nodes, at))
        arena = [dict(n) for n in nodes]
        payload = dict(arena[at]["payload"])
        kids = list(payload.get("children", []))
        if len(kids) == 2:
            kids[0], kids[1] = kids[1], kids[0]
        payload["children"] = kids
        arena[at] = {"kind": arena[at]["kind"], "payload": payload}
        return {
            "rewrites": 1,
            "root_index": at,
            "root": arena[at],
            "arena": arena,
        }


def _vars() -> tuple[Var, Var]:
    return Var("a", "f64"), Var("b", "f64")


def _comm_graph() -> tuple[HandlEngine, int]:
    """An engine with ``add(a, b)`` interned; returns (eg, root)."""
    eg = HandlEngine(peer=_SwapPeer())
    a, b = _vars()
    return eg, eg.add_term(Op.make("add", a, b))


_COMM = next(r for r in ALL_RULES if r.name == "comm_add")
_ASSOC = next(r for r in ALL_RULES if r.name == "assoc_add")


# ---------------------------------------------------------------------------
#  Peer resolution — the honest-skip surface
# ---------------------------------------------------------------------------


def test_peer_unavailable_names_the_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No extension, no CLI → HandlPeerUnavailable with instructions."""
    monkeypatch.setattr(handl_engine, "_ext", None)
    monkeypatch.setenv("MLOPT_HANDL_PEER_BIN", str(tmp_path / "nope"))
    with pytest.raises(HandlPeerUnavailable, match="maturin develop"):
        HandlPeer(fragments=tmp_path)


def test_peer_unavailable_without_fragment_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing fragment directory declines before surface checks."""
    monkeypatch.setattr(handl_engine, "_ext", object())
    monkeypatch.delenv("MLOPT_VOCAB_DIR", raising=False)
    with pytest.raises(HandlPeerUnavailable, match="fragment"):
        HandlPeer(fragments=tmp_path / "absent")


def test_handl_engine_requires_a_peer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``HandlEngine()`` with no peer built refuses at construction."""
    monkeypatch.setattr(handl_engine, "_ext", None)
    monkeypatch.setenv("MLOPT_HANDL_PEER_BIN", str(tmp_path / "nope"))
    with pytest.raises(HandlPeerUnavailable):
        HandlEngine(fragments=tmp_path)


def test_peer_cli_surface_parses_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The CLI fallback runs the binary and parses its JSON."""
    monkeypatch.setattr(handl_engine, "_ext", None)
    payload = {
        "rewrites": 1,
        "root_index": 0,
        "root": {"kind": "x", "payload": {}},
        "arena": [{"kind": "x", "payload": {}}],
    }
    bin_path = tmp_path / "rewrite"
    bin_path.write_text(f"#!/bin/sh\necho '{json.dumps(payload)}'\n")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("MLOPT_HANDL_PEER_BIN", str(bin_path))
    frag = tmp_path / "frag"
    frag.mkdir()

    peer = HandlPeer(fragments=frag)
    assert peer.surface == "cli"
    out = peer.apply_rule("transform.mlopt.comm_add", [], 0)
    assert out == payload
    with pytest.raises(HandlPeerUnavailable, match="extension"):
        peer.transform_ids()


def test_peer_cli_decline_is_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A nonzero CLI exit is a decline (``None``), never an exception."""
    monkeypatch.setattr(handl_engine, "_ext", None)
    bin_path = tmp_path / "rewrite"
    bin_path.write_text("#!/bin/sh\nexit 3\n")
    bin_path.chmod(bin_path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("MLOPT_HANDL_PEER_BIN", str(bin_path))
    frag = tmp_path / "frag"
    frag.mkdir()
    peer = HandlPeer(fragments=frag)
    assert peer.apply_rule("transform.mlopt.x", [], 0) is None


# ---------------------------------------------------------------------------
#  Term <-> AST helpers
# ---------------------------------------------------------------------------


def test_term_to_ast_is_post_order() -> None:
    """Leaves precede their parents; the root is the last index."""
    a, b = _vars()
    nodes, root = term_to_ast(Op.make("add", a, Op.make("mul", a, b)))
    assert root == len(nodes) - 1
    assert [n["kind"] for n in nodes] == ["a", "a", "b", "mul", "add"]
    assert nodes[-1]["payload"]["children"] == [0, 3]


def test_render_term_matches_handl_shape() -> None:
    a, b = _vars()
    assert render_term(Op.make("add", a, b)) == {
        "kind": "add",
        "children": [
            {"kind": "a", "children": []},
            {"kind": "b", "children": []},
        ],
    }


def test_is_structural_rhs_filters_leaf_rhs() -> None:
    """Ops/metavars only — a Const/Var rhs is not comparable yet."""
    assert is_structural_rhs(Op.make("add", "a", "b"))
    assert is_structural_rhs("a")  # a bare metavar (root-eliminating)
    assert not is_structural_rhs(Op.make("add", "a", Const(1.0)))
    assert not is_structural_rhs(Var("x", "f64"))


# ---------------------------------------------------------------------------
#  The differential engine
# ---------------------------------------------------------------------------


def test_engine_agrees_and_records_parity() -> None:
    """comm_add fires; the fake peer's swap agrees; parity counts it."""
    eg, root = _comm_graph()
    stats = eg.run([_COMM], root)
    assert stats["handl_parity"]["agree"] == 1
    assert stats["handl_parity"]["disagreed"] == []
    assert stats["handl_parity"]["surface"] == "fake"
    assert eg.engine_name == "handl"  # what search() records
    # the peer saw transform.mlopt.comm_add at the root
    rule_id, _nodes, at = eg._peer.calls[0]
    assert rule_id == "transform.mlopt.comm_add"
    assert at == 2  # add(a, b) flattens to [a, b, add] — root last


def test_engine_disagreement_refuses_loudly() -> None:
    """A peer returning the input unchanged disagrees → raise."""
    eg = HandlEngine(peer=_IdentityPeer())
    a, b = _vars()
    root = eg.add_term(Op.make("add", a, b))
    with pytest.raises(HandlDisagreementError, match="comm_add"):
        eg.run([_COMM], root)
    detail = eg.handl_parity["disagreed"][0]
    assert detail["rule"] == "comm_add"
    assert detail["mlopt_rhs"]["kind"] == "add"


def test_engine_decline_is_counted_not_fatal() -> None:
    """A peer that cannot fire counts ``declined`` — parity, not error."""
    eg = HandlEngine(peer=_IdentityPeer(decline=True))
    a, b = _vars()
    root = eg.add_term(Op.make("add", a, b))
    stats = eg.run([_COMM], root)
    assert stats["handl_parity"]["declined"] == 1
    assert stats["handl_parity"]["agree"] == 0


def test_engine_skips_leaf_rhs_laws() -> None:
    """A law whose rhs carries a leaf is counted ``skipped``."""
    leaf_law = Rewrite(
        "zz_leaf_rhs",
        Op.make("add", "a", "b"),
        Op.make("mul", "a", Const(2.0)),
    )
    eg = HandlEngine(peer=_SwapPeer())
    a, b = _vars()
    root = eg.add_term(Op.make("add", a, b))
    stats = eg.run([leaf_law], root)
    assert stats["handl_parity"]["skipped"] == 1
    assert stats["handl_parity"]["peer_calls"] == 0


def test_engine_skips_derive_rules() -> None:
    """A ``derive`` hook mints bindings handl cannot see → ``skipped``."""
    derived = Rewrite(
        "zz_derived",
        Op.make("add", "a", "b"),
        Op.make("mul", "a", "b"),
        derive=lambda bound: {"$leaf:extra": Const(0.0)},
    )
    eg = HandlEngine(peer=_SwapPeer())
    a, b = _vars()
    root = eg.add_term(Op.make("add", a, b))
    stats = eg.run([derived], root)
    assert stats["handl_parity"]["skipped"] == 1
    assert stats["handl_parity"]["peer_calls"] == 0


def test_peer_cli_spawn_failure_is_a_decline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bin that exists but cannot exec declines via OSError."""
    monkeypatch.setattr(handl_engine, "_ext", None)
    bin_path = tmp_path / "rewrite"
    bin_path.write_text("#!/bin/sh\nexit 0\n")  # not executable
    frag = tmp_path / "frag"
    frag.mkdir()
    monkeypatch.setenv("MLOPT_HANDL_PEER_BIN", str(bin_path))
    peer = HandlPeer(fragments=frag)
    assert peer.apply_rule("transform.mlopt.x", [], 0) is None


def test_engine_counts_unresolvable_applications() -> None:
    """An application with no recorded Rewrite counts ``unresolved``."""
    eg, _root = _comm_graph()
    eg._applications.append({"rule": "ghost", "subst": {}})
    eg.check_pending()
    assert eg.handl_parity["unresolved"] == 1


def test_engine_memoizes_repeat_firings() -> None:
    """The same (rule, lhs) firing checks once — peer_calls stays 1."""
    eg, root = _comm_graph()
    eg.run([_COMM], root)
    eg.check_pending()  # no new applications — cursor holds
    assert eg.handl_parity["agree"] == 1
    peer = eg._peer
    assert len(peer.calls) == 1


def test_apply_rule_outside_run_is_checkable() -> None:
    """Direct ``apply_rule`` firings are covered by ``check_pending``."""
    eg, root = _comm_graph()
    assert eg.apply_rule(_COMM, root)
    eg.check_pending()
    assert eg.handl_parity["agree"] == 1


def test_truncation_level_one_reports_untracked() -> None:
    """Level-1 e-graphs record no applications — parity says so."""
    eg = HandlEngine(peer=_SwapPeer(), truncation_level=1)
    a, b = _vars()
    root = eg.add_term(Op.make("add", a, b))
    stats = eg.run([_COMM], root)
    assert stats["handl_parity"]["tracked"] is False
    assert stats["handl_parity"]["agree"] == 0


def test_resolve_engine_spellings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``resolve_engine`` maps the plan's ``engine="handl"`` spelling."""
    assert isinstance(resolve_engine(None), EGraph)
    assert isinstance(resolve_engine("python"), EGraph)

    sentinel = HandlEngine(peer=_SwapPeer())
    monkeypatch.setattr(
        handl_engine, "HandlEngine", lambda *a, **k: sentinel
    )
    assert resolve_engine("handl") is sentinel
    assert resolve_engine(sentinel) is sentinel  # passthrough


# ---------------------------------------------------------------------------
#  End-to-end — only where the peer is really built
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_handl_engine_saturates_and_agrees() -> None:
    """Real peer: assoc_add fires on both engines identically.

    Needs the maturin extension or the CLI binary (see
    ``test_handl_peer.py`` for the build); skips without either.
    Uses the default resolution — ``HandlEngine()`` builds a
    :class:`HandlPeer` over the repo ``vocab/`` projection, the same
    directory ``pipeline.propose`` consumes.
    """
    try:
        eg = HandlEngine()
    except HandlPeerUnavailable:
        pytest.skip("handl peer not built (see test_handl_peer.py)")
    assert eg._peer.fragments == _VOCAB
    # the extension surface also answers transform_ids
    if eg._peer.surface == "extension":
        assert "transform.mlopt.assoc_add" in eg._peer.transform_ids()
        # an unknown rule id is a decline, not a crash
        assert (
            eg._peer.apply_rule("transform.mlopt.nope", [], 0) is None
        )
    a, b, c = (
        Var("a", "f64"),
        Var("b", "f64"),
        Var("c", "f64"),
    )
    root = eg.add_term(Op.make("add", a, Op.make("add", b, c)))
    stats = eg.run([_ASSOC], root)
    parity = stats["handl_parity"]
    assert parity["disagreed"] == []
    assert parity["agree"] >= 1
