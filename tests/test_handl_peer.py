"""The handl peer's call path — mlopt drives handl's rewrite engine.

This is plan 0030 step 2's proof.  The peer (`packages/mlopt-handl-peer`) is a PyO3
extension, `mlopt_handl_peer`, built with maturin; these tests drive it
through the **extension** when it is importable:

    cd packages/mlopt-handl-peer
    VIRTUAL_ENV=../../.venv maturin develop \\
        --target-dir ~/GitHub/handl-lang/host/target

The same crate also builds a CLI over the same library code
(`src/bin/rewrite.rs`), and that is the **fallback** surface — a checkout
with no maturin build (and no `mlopt_handl_peer` on `sys.path`) still
exercises the call path through the binary:

    cd packages/mlopt-handl-peer
    PYO3_PYTHON=$(git rev-parse --show-toplevel)/.venv/bin/python \\
        CARGO_TARGET_DIR=~/GitHub/handl-lang/host/target \\
        cargo build --bin rewrite

They **skip** when neither surface is available — the extension is a
spike artifact, not part of the wheel:

    pytestmark = skipif(no extension and no CLI binary, or no fragments)

`PYO3_PYTHON` is load-bearing on this host: `python3` on `PATH` is 3.14,
and a raw `cargo build` binds against it, emitting an extern `Py_TYPE`
symbol (pyo3-ffi makes `Py_TYPE` extern only under `Py_3_14`) that the
3.13 runtime does not export — the import then fails with
`undefined symbol: Py_TYPE`.  maturin picks the venv interpreter itself.

What is proven here: mlopt calls handl, handl loads mlopt's fragment
directory as a `Language`, applies one of mlopt's laws, and returns the
term — the same term mlopt's own engine produces.  What is *not*
proven: any tensor crossing (ADR 0011 keeps those out of v1 on purpose).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op, Var
from mlopt_core.laws import ALL_RULES

_REPO = Path(__file__).resolve().parents[1]
_BIN = Path(
    os.environ.get(
        "MLOPT_HANDL_PEER_BIN",
        Path.home() / "GitHub/handl-lang/host/target/debug/rewrite",
    )
)
_FRAGMENTS = _REPO / "tests" / "fixtures" / "handl-spike" / "fragments"

try:  # the shipping surface — absent until maturin builds it
    import mlopt_handl_peer as _ext
except ImportError:  # pragma: no cover - depends on the local build
    _ext = None

pytestmark = [
    pytest.mark.e2e,  # crosses into the compiled handl engine
    pytest.mark.skipif(
        (_ext is None and not _BIN.exists()) or not _FRAGMENTS.is_dir(),
        reason=(
            "handl peer (extension or CLI) or fragment dir not available "
            "(see the docstring)"
        ),
    ),
]


def _ast(*nodes: tuple[str, list[int]]) -> str:
    return json.dumps(
        [{"kind": k, "payload": {"children": list(c)}} for k, c in nodes]
    )


def _run(*args: str) -> str:
    """Rewrite through the peer: the extension if importable, else the CLI.

    *args* is the CLI's spelling — ``(rule_id, at, ast_json)``.  Both
    surfaces call the same Rust library, so the result is identical; the
    extension is preferred and the CLI is the no-maturin fallback.
    """
    rule_id, at, ast = args
    if _ext is not None:
        return _ext.apply_rule(str(_FRAGMENTS), ast, int(at), rule_id)
    out = subprocess.run(
        [str(_BIN), str(_FRAGMENTS), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_the_extension_is_the_preferred_surface() -> None:
    """The PyO3 extension is importable and reports the fragment's laws.

    Skips — without failing — when only the CLI is built, so a fresh
    checkout stays green; the fallback still exercises the call path.
    """
    if _ext is None:
        pytest.skip("extension not installed; using the CLI fallback")
    ids = _ext.language_transform_ids(str(_FRAGMENTS))
    assert "transform.mlopt.assoc_add" in ids


def test_the_peer_rewrites_with_a_mlopt_law() -> None:
    """handl applies `assoc_add` and returns add(add(a,b), c)."""
    # add(a, add(b, c))
    ast = _ast(("a", []), ("b", []), ("c", []), ("add", [1, 2]), ("add", [0, 3]))
    result = json.loads(_run("transform.mlopt.assoc_add", "4", ast))

    assert result["rewrites"] == 1, "exactly one rewrite"
    arena, root = result["arena"], result["root"]
    assert root["kind"] == "add"
    kids = root["payload"]["children"]
    left = arena[kids[0]]
    assert left["kind"] == "add", "left child is the new add(a, b)"
    assert [arena[i]["kind"] for i in left["payload"]["children"]] == ["a", "b"]
    assert arena[kids[1]]["kind"] == "c", "right child is c"


def test_the_peer_agrees_with_mlopts_own_engine() -> None:
    """The same term, both engines — the two must agree."""
    ast = _ast(("a", []), ("b", []), ("c", []), ("add", [1, 2]), ("add", [0, 3]))
    handl_root = json.loads(_run("transform.mlopt.assoc_add", "4", ast))["root"]
    assert handl_root["kind"] == "add"

    law = next(r for r in ALL_RULES if r.name == "assoc_add")
    a, b, c = Var("a", "f64"), Var("b", "f64"), Var("c", "f64")
    graph = EGraph()
    root = graph.add_term(Op.make("add", a, Op.make("add", b, c)))
    assert graph.apply_rule(law, root), "mlopt's own engine fires the rule"
    expected = Op.make("add", Op.make("add", a, b), c)
    assert graph.find(graph.add_term(expected)) == graph.find(root), (
        "mlopt's engine agrees: add(add(a,b),c) is the rewritten member"
    )


def test_the_differential_reports_no_disagreement() -> None:
    """mlopt's engine and handl's agree wherever both fire.

    The probe (`bench/drivers/audit/handl_differential.py`) synthesizes terms per
    law — attribute metavars over a small candidate pool, so guarded laws
    are exercised on *both* sides of their guard — rewrites them with
    both engines and compares the results.  This pins the *measured*
    state: 48 agreeing applications, 0 disagreements, and — the
    Phase-B decline-parity arm — **0 over-fires** (mlopt declined
    where handl fired) across 40 both-declined instantiations and 76
    explicit ``!Native`` refusals.  The laws it cannot compare are
    reported by the probe with their reason (leaf rhs, guard raised)
    -- none of which is a disagreement.  Root-eliminating laws
    (``id_add``, ``id_mul``, ``double_neg``, ``transpose_noop``,
    ``chunk_single``) are *expressible* via handl's ``!Replace``
    (handl ADR-0127) plus the new attr guards — see
    `test_root_eliminating_laws_agree_on_satisfying_terms`.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "handl_differential",
        _REPO / "bench" / "drivers" / "audit" / "handl_differential.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = mod.main([])
    report = buf.getvalue()
    assert "DISAGREE           : 0" in report, report
    assert "OVERFIRE (mlopt no, handl yes): 0" in report, report
    assert "agree              : 48" in report, report
    assert code == 0, report


def test_root_eliminating_laws_agree_on_satisfying_terms() -> None:
    """Guarded root-eliminating laws agree *or refuse* — never over-fire.

    ``chunk_single``'s ``attr-eq CK 1`` guard is attr-decidable, so the
    bridge emits a real ``!Replace`` guarded by ``AttrEq`` clauses:
    on a satisfying term both engines return the subterm.  The probe's
    generic term cannot satisfy ``transpose_noop``'s ``axes-noop``
    guard — that predicate needs mlopt's shape oracle, which handl's
    AST does not carry — so the bridge refuses it explicitly as
    ``!Native "mlopt.law.transpose_noop"`` and handl declines *even
    when mlopt fires*.  That is the honest gap the refusal names
    (Phase-B report §4), pinned here so a silent regression back to an
    unguarded ``!Replace`` cannot pass.
    """
    from mlopt_core.egraph import EGraph
    from mlopt_core.ir import Op, TensorType, Var
    from kit_discovery.handl_engine import term_to_ast

    fragments_all = _REPO / "tests" / "fixtures" / "handl-spike" / "fragments-all"
    assert fragments_all.is_dir(), "the full bank fragment dir is required"

    def _apply(name: str, term: object):
        """Run the peer on *term*; ``None`` on decline/refusal."""
        nodes, at = term_to_ast(term)
        ast = json.dumps(nodes)
        if _ext is not None:
            try:
                return json.loads(
                    _ext.apply_rule(
                        str(fragments_all), ast, at,
                        f"transform.mlopt.{name}",
                    )
                )
            except Exception:
                return None
        proc = subprocess.run(
            [str(_BIN), str(fragments_all), f"transform.mlopt.{name}",
             str(at), ast],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            return None
        return json.loads(proc.stdout)

    # chunk(u, chunks=1, …) -> u — the guard holds, both engines agree.
    term = Op.make(
        "chunk", Var("u", TensorType((4,))), chunks=1, dim=0, index=0
    )
    law = next(r for r in ALL_RULES if r.name == "chunk_single")
    graph = EGraph()
    assert graph.apply_rule(law, graph.add_term(term)), (
        "mlopt's engine must fire chunk_single on a satisfying term"
    )
    out = _apply("chunk_single", term)
    assert out is not None, "handl must not refuse the attr-guarded law"
    root = out["arena"][out["root_index"]]
    assert root["kind"] == "u", f"chunk_single: got {root}"

    # chunk(u, chunks=3, …) — mlopt declines; handl must decline too
    # (the ``AttrEq chunks == 1`` clause is the guard that used to be
    # missing — the Phase-B over-fire).
    bad = Op.make(
        "chunk", Var("u", TensorType((4,))), chunks=3, dim=0, index=0
    )
    graph = EGraph()
    assert not graph.apply_rule(law, graph.add_term(bad)), (
        "mlopt's engine must decline chunk_single when chunks != 1"
    )
    assert _apply("chunk_single", bad) is None, (
        "handl fired chunk_single where mlopt declined — over-fire"
    )

    # transpose(u, 0, 0) -> u — mlopt's ``axes-noop`` guard holds, but
    # the predicate is shape-only: handl refuses the law outright.
    term = Op.make(
        "transpose", Var("u", TensorType((4, 4))), dim0=0, dim1=0
    )
    law = next(r for r in ALL_RULES if r.name == "transpose_noop")
    graph = EGraph()
    assert graph.apply_rule(law, graph.add_term(term)), (
        "mlopt's engine must fire transpose_noop on a noop transpose"
    )
    assert _apply("transpose_noop", term) is None, (
        "transpose_noop must stay a !Native refusal — its guard is "
        "shape-only and handl cannot express it"
    )
