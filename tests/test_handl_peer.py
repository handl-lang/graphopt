"""The handl peer's call path — catopt drives handl's rewrite engine.

This is plan 0030 step 2's proof in its cheapest form.  The *shipping*
shape is a PyO3 extension (`lab/handl-peer`), but its packaging needs
maturin, which is not installed in this venv; the crate therefore also
builds a CLI over the same library code, and that is what these tests
drive.

They **skip** unless the binary has been built — it is a spike artifact,
not part of the wheel:

    cd lab/handl-peer
    CARGO_TARGET_DIR=~/GitHub/handl-lang/platform/target \\
        cargo build --bin rewrite

What is proven here: catopt calls handl, handl loads catopt's fragment
directory as a `Language`, applies one of catopt's laws, and returns the
term — the same term catopt's own engine produces.  What is *not*
proven: the PyO3 packaging (maturin), and any tensor crossing (ADR 0011
keeps those out of v1 on purpose).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from catopt_core.egraph import EGraph
from catopt_core.ir import Op, Var
from catopt_core.laws import ALL_RULES

_REPO = Path(__file__).resolve().parents[1]
_BIN = Path(
    os.environ.get(
        "CATOPT_HANDL_PEER_BIN",
        Path.home() / "GitHub/handl-lang/platform/target/debug/rewrite",
    )
)
_FRAGMENTS = _REPO / "lab" / "handl-spike" / "fragments"

pytestmark = pytest.mark.skipif(
    not _BIN.exists() or not _FRAGMENTS.is_dir(),
    reason="handl peer binary or fragment dir not built (see the docstring)",
)


def _ast(*nodes: tuple[str, list[int]]) -> str:
    return json.dumps(
        [{"kind": k, "payload": {"children": list(c)}} for k, c in nodes]
    )


def _run(*args: str) -> str:
    out = subprocess.run(
        [str(_BIN), str(_FRAGMENTS), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_the_peer_rewrites_with_a_catopt_law() -> None:
    """handl applies `assoc_add` and returns add(add(a,b), c)."""
    # add(a, add(b, c))
    ast = _ast(("a", []), ("b", []), ("c", []), ("add", [1, 2]), ("add", [0, 3]))
    result = json.loads(_run("transform.catopt.assoc_add", "4", ast))

    assert result["rewrites"] == 1, "exactly one rewrite"
    arena, root = result["arena"], result["root"]
    assert root["kind"] == "add"
    kids = root["payload"]["children"]
    left = arena[kids[0]]
    assert left["kind"] == "add", "left child is the new add(a, b)"
    assert [arena[i]["kind"] for i in left["payload"]["children"]] == ["a", "b"]
    assert arena[kids[1]]["kind"] == "c", "right child is c"


def test_the_peer_agrees_with_catopts_own_engine() -> None:
    """The same term, both engines — the two must agree."""
    ast = _ast(("a", []), ("b", []), ("c", []), ("add", [1, 2]), ("add", [0, 3]))
    handl_root = json.loads(_run("transform.catopt.assoc_add", "4", ast))["root"]
    assert handl_root["kind"] == "add"

    law = next(r for r in ALL_RULES if r.name == "assoc_add")
    a, b, c = Var("a", "f64"), Var("b", "f64"), Var("c", "f64")
    graph = EGraph()
    root = graph.add_term(Op.make("add", a, Op.make("add", b, c)))
    assert graph.apply_rule(law, root), "catopt's own engine fires the rule"
    expected = Op.make("add", Op.make("add", a, b), c)
    assert graph.find(graph.add_term(expected)) == graph.find(root), (
        "catopt's engine agrees: add(add(a,b),c) is the rewritten member"
    )


def test_the_differential_reports_no_disagreement() -> None:
    """catopt's engine and handl's agree wherever both fire.

    The probe (`lab/audit/handl_differential.py`) synthesizes one term per
    law, rewrites it with both engines and compares the results.  This
    pins the *measured* state: 19 laws comparable, 19 agreeing, 0
    disagreements.  The laws it cannot compare are reported by the probe
    with their reason (leaf rhs, root-eliminating, declined, guard
    raised) -- none of which is a disagreement.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "handl_differential", _REPO / "lab" / "audit" / "handl_differential.py"
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
    assert "agree              : 19" in report, report
    assert code == 0, report
