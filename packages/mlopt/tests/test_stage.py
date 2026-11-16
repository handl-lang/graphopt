"""``mlopt stage`` protocol tests — the kit-mlopt JSON endpoint.

Each stage is exercised through ``mlopt.cli.main(["stage", name])``
with stdin/stdout captured — the same code path ``python -m mlopt
stage`` takes.  The fixture is the tiny ``.hnl`` program from
``test_hnl_source``: a real lift → search → verify chain over the
shipped machinery with no model downloads.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

import mlopt.models.catalog  # noqa: F401 — populates the registry
from mlopt.cli import main

try:
    import torch  # noqa: F401 — the pipeline's sink is torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover — torch is a hard dep
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(not _HAS_TORCH, reason="torch required")


@pytest.fixture()
def tiny_hnl(tmp_path):
    """A ``do``-block program: one ``Param`` weight, one ``Var`` input."""
    path = tmp_path / "tiny.hnl"
    path.write_text(
        "--- tiny test program\n"
        'from "host-tensor" import tensor\n'
        "do {\n"
        "  w :: tensor.reshape(tensor.from_list([1, 2, 3, 4, 5, 6]), [2, 3])\n"
        "  tensor.matmul(x, w)\n"
        "}\n"
    )
    return path


def _ref(path) -> str:
    return f"hnl:{path}"


def _lift_input(path, **extra):
    rec = {
        "model": _ref(path),
        "adapter_opts": {"leaves": {"x": [4, 2]}},
    }
    rec.update(extra)
    return rec


def _run(capsys, monkeypatch, name, inp):  # noqa: ANN001, ANN202
    """Drive ``mlopt stage <name>`` in-process; return (code, record, stderr)."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(inp)))
    code = main(["stage", name])
    captured = capsys.readouterr()
    out = captured.out.strip()
    rec = json.loads(out) if out else None
    return code, rec, captured.err


def _pipeline(capsys, monkeypatch, path):  # noqa: ANN001, ANN202
    """lift → search → verify over the tiny .hnl; return the three records."""
    code, lift, _ = _run(capsys, monkeypatch, "lift", _lift_input(path))
    assert code == 0 and lift["ok"] is True
    code, search, _ = _run(capsys, monkeypatch, "search", lift)
    assert code == 0 and search["ok"] is True
    code, verify, _ = _run(capsys, monkeypatch, "verify", search)
    return lift, search, verify, code


# --- lift --------------------------------------------------------------------


def test_lift_record_shape(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    code, rec, err = _run(capsys, monkeypatch, "lift", _lift_input(tiny_hnl))
    assert code == 0, err
    assert rec["stage"] == "lift"
    assert rec["model"] == _ref(tiny_hnl)
    assert rec["adapter"] == "hnl"
    assert rec["backend"] == "torch"
    assert rec["ok"] is True
    # The lifted IR is real term data: matmul(x, w) with typed leaves,
    # spelled in the v3 hnl_codec Term-JSON envelope (``Op``/``Arr``).
    root = rec["ir"]["root"]["Op"]
    assert root["name"] == "matmul"
    x_leaf = root["args"][0]["Arr"]["value"]
    assert x_leaf["__mlopt_leaf__"] == "var"
    assert x_leaf["name"] == "x"
    assert x_leaf["shape"] == [4, 2]
    w_leaf = root["args"][1]["Arr"]["value"]
    assert w_leaf["__mlopt_leaf__"] == "param"
    assert w_leaf["name"] == "w"
    w_param = rec["ir"]["params"]["w"]["Arr"]["value"]
    assert w_param["__mlopt_leaf__"] == "param"
    assert w_param["shape"] == [2, 3]
    assert rec["leaf_names"] == ["w"]
    assert rec["leaf_source"] == "synthetic"
    assert any("re-materialised" in n for n in rec["notes"])


def test_lift_bare_ref_declines_unbound(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    # The tiny.hnl Var is shapeless, so a bare-string lift (no
    # adapter_opts.leaves) must decline honestly naming the unbound
    # leaf — not crash, not guess a shape.
    code, rec, err = _run(capsys, monkeypatch, "lift", _ref(tiny_hnl))
    assert code == 1
    assert rec["ok"] is False
    assert "unbound" in rec["error"]
    assert "unbound" in err


def test_lift_unknown_ref(capsys, monkeypatch) -> None:  # noqa: ANN001
    code, rec, err = _run(capsys, monkeypatch, "lift", "nosuch:model")
    assert code == 1
    assert rec["ok"] is False
    assert "no adapter" in err


def test_lift_missing_file(capsys, monkeypatch) -> None:  # noqa: ANN001
    code, rec, err = _run(capsys, monkeypatch, "lift", "hnl:/nonexistent/x.hnl")
    assert code == 1
    assert rec["ok"] is False
    assert "cannot read" in err


# --- search ------------------------------------------------------------------


def test_search_threads_lift_record(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    _lift, search, _verify, _code = _pipeline(capsys, monkeypatch, tiny_hnl)
    assert search["stage"] == "search"
    assert search["model"] == _ref(tiny_hnl)
    assert search["term"]["Op"]["name"] == "matmul"
    assert isinstance(search["n_enodes"], int)
    assert "rule_fires" in search["stats"] or "iterations" in search["stats"]


def test_search_requires_ir(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    code, rec, err = _run(
        capsys, monkeypatch, "search", {"model": _ref(tiny_hnl), "adapter_opts": {}}
    )
    assert code == 1
    assert rec["ok"] is False
    assert "lift" in err


def test_search_rejects_non_record(capsys, monkeypatch) -> None:  # noqa: ANN001
    code, rec, _err = _run(capsys, monkeypatch, "search", "not-a-record")
    assert code == 1
    assert rec["ok"] is False


# --- verify ------------------------------------------------------------------


def test_verify_passes_on_searched_term(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    _lift, _search, verify, code = _pipeline(capsys, monkeypatch, tiny_hnl)
    assert code == 0
    assert verify["stage"] == "verify"
    assert verify["ok"] is True
    assert verify["verified"]["passed"] is True
    assert verify["verified"]["max_abs"] == 0.0


def test_verify_requires_term(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    _code, lift, _err = _run(capsys, monkeypatch, "lift", _lift_input(tiny_hnl))
    code, rec, err = _run(capsys, monkeypatch, "verify", lift)
    assert code == 1
    assert rec["ok"] is False
    assert "search" in err


def test_verify_declined_exits_nonzero(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    """A searched term that is NOT equivalent must fail the real gate."""
    _lift, search, _verify, _code = _pipeline(capsys, monkeypatch, tiny_hnl)
    x = {"var": "x", "shape": [4, 2]}
    w = {"param": "w", "shape": [2, 3]}
    # The legacy ``term_to_data`` spelling — exercises the v≤2 decode
    # path inside a v3-stamped record (payloads are shape-dispatched).
    mm = lambda a, b: {"op": "matmul", "args": [a, b], "attrs": {}}  # noqa: E731
    # matmul(x,w)² elementwise — runs, same shape, wrong values.
    search["term"] = {"op": "mul", "args": [mm(x, w), mm(x, w)], "attrs": {}}
    code, rec, _err = _run(capsys, monkeypatch, "verify", search)
    assert code == 1
    assert rec["ok"] is False
    assert rec["verified"]["passed"] is False
    assert rec["verified"]["max_abs"] > 0


# --- deliver -----------------------------------------------------------------


def _deliver_input(verify_rec, out):
    rec = dict(verify_rec)
    rec["out"] = str(out)
    return rec


def test_deliver_writes_verified_hnl_artifact(tiny_hnl, tmp_path, capsys, monkeypatch) -> None:  # noqa: ANN001
    _lift, _search, verify, code = _pipeline(capsys, monkeypatch, tiny_hnl)
    assert code == 0 and verify["verified"]["passed"] is True
    out = tmp_path / "delivered.hnl"
    code, rec, err = _run(capsys, monkeypatch, "deliver", _deliver_input(verify, out))
    assert code == 0, err
    assert rec["stage"] == "deliver"
    assert rec["ok"] is True
    assert rec["delivered"] is True
    assert rec["verified"]["passed"] is True
    assert rec["artifact"]["path"] == str(out)
    assert rec["artifact"]["format"] == "hnl"
    # F3: the record says whether the emitted program actually ran —
    # the verify verdict covers the term, not the .hnl semantics.
    assert "artifact_executed" in rec
    assert isinstance(rec["artifact_executed"], bool)
    if rec["artifact_executed"]:
        assert any("smoke-executed" in n for n in rec["notes"])
    else:
        assert any("NOT executed" in n for n in rec["notes"])
    src = out.read_text()
    assert 'from "host-tensor" import tensor' in src
    assert "tensor.matmul" in src
    assert "tensor.from_list" in src  # leaves bound with real values


def test_deliver_refuses_unverified_claim(tiny_hnl, tmp_path, capsys, monkeypatch) -> None:  # noqa: ANN001
    """A record whose verify report did not pass is refused — no artifact."""
    _lift, _search, verify, _code = _pipeline(capsys, monkeypatch, tiny_hnl)
    verify["verified"]["passed"] = False
    verify["verified"]["max_abs"] = 1.0
    out = tmp_path / "nope.hnl"
    code, rec, _err = _run(capsys, monkeypatch, "deliver", _deliver_input(verify, out))
    assert code == 1
    assert rec["ok"] is False
    assert rec["delivered"] is False
    assert "did not pass" in rec["error"]
    assert not out.exists()


def test_deliver_rejects_tampered_term(tiny_hnl, tmp_path, capsys, monkeypatch) -> None:  # noqa: ANN001
    """A forged ``verified.passed`` claim cannot ship an unverified term.

    The record claims verify passed but carries a term that is NOT
    equivalent — deliver re-runs the real ``lower(verify=True)`` gate,
    which declines, and no artifact is written.
    """
    _lift, search, verify, _code = _pipeline(capsys, monkeypatch, tiny_hnl)
    x = {"var": "x", "shape": [4, 2]}
    w = {"param": "w", "shape": [2, 3]}
    mm = lambda a, b: {"op": "matmul", "args": [a, b], "attrs": {}}  # noqa: E731
    verify["term"] = {
        "op": "mul",
        "args": [mm(x, w), mm(x, w)],
        "attrs": {},
    }  # legacy spelling, see above
    # The forged claim: verified.passed stays True — deliver must not
    # trust it; the re-run gate is the arbiter.
    assert verify["verified"]["passed"] is True
    out = tmp_path / "forged.hnl"
    code, rec, _err = _run(capsys, monkeypatch, "deliver", _deliver_input(verify, out))
    assert code == 1
    assert rec["ok"] is False
    assert rec["delivered"] is False
    assert "declined" in rec["error"]
    assert rec["verified"]["passed"] is False
    assert not out.exists()


def test_deliver_hnl_run_rejection_declines(tiny_hnl, tmp_path, capsys, monkeypatch) -> None:  # noqa: ANN001
    """A program hnl-run rejects must not ship — decline, no artifact."""
    import mlopt.emit

    _lift, _search, verify, code = _pipeline(capsys, monkeypatch, tiny_hnl)
    assert code == 0 and verify["verified"]["passed"] is True
    out = tmp_path / "rejected.hnl"

    def _boom(self, src, repeat=1):  # noqa: ANN001, ANN202
        raise RuntimeError("kit refused the program")

    monkeypatch.setattr(mlopt.emit, "find_hnl_run", lambda: sys.executable)
    monkeypatch.setattr(mlopt.emit.KitRunner, "run", _boom)
    code, rec, _err = _run(capsys, monkeypatch, "deliver", _deliver_input(verify, out))
    assert code == 1
    assert rec["ok"] is False
    assert rec["delivered"] is False
    assert rec["artifact_executed"] is False
    assert "hnl-run rejected" in rec["error"]
    assert not out.exists()


def test_deliver_requires_verify_report(tiny_hnl, tmp_path, capsys, monkeypatch) -> None:  # noqa: ANN001
    _lift, search, _verify, _code = _pipeline(capsys, monkeypatch, tiny_hnl)
    out = tmp_path / "x.hnl"
    code, rec, err = _run(capsys, monkeypatch, "deliver", _deliver_input(search, out))
    assert code == 1
    assert rec["ok"] is False
    assert "verify" in err
    assert not out.exists()


def test_deliver_requires_out(tiny_hnl, tmp_path, capsys, monkeypatch) -> None:  # noqa: ANN001
    _lift, _search, verify, _code = _pipeline(capsys, monkeypatch, tiny_hnl)
    code, rec, err = _run(capsys, monkeypatch, "deliver", verify)
    assert code == 1
    assert rec["ok"] is False
    assert "'out'" in err


def test_deliver_rejects_non_record(capsys, monkeypatch) -> None:  # noqa: ANN001
    code, rec, _err = _run(capsys, monkeypatch, "deliver", "not-a-record")
    assert code == 1
    assert rec["ok"] is False


def test_deliver_unwritable_out_declines(tiny_hnl, tmp_path, capsys, monkeypatch) -> None:  # noqa: ANN001
    _lift, _search, verify, _code = _pipeline(capsys, monkeypatch, tiny_hnl)
    out = tmp_path / "missing-dir" / "x.hnl"
    code, rec, _err = _run(capsys, monkeypatch, "deliver", _deliver_input(verify, out))
    assert code == 1
    assert rec["ok"] is False
    assert rec["delivered"] is False
    assert "cannot write" in rec["error"]


def test_deliver_input_arity_mismatch_declines(tiny_hnl, tmp_path, capsys, monkeypatch) -> None:  # noqa: ANN001
    """F10: ``ir.inputs`` vs ``spec.inputs`` arity mismatch is a named
    decline, never a silent ``zip`` truncation."""
    _lift, _search, verify, code = _pipeline(capsys, monkeypatch, tiny_hnl)
    assert code == 0 and verify["verified"]["passed"] is True
    # Forge a second declared input the spec cannot feed.
    verify["ir"]["inputs"].append({"var": "y", "shape": [1]})
    out = tmp_path / "arity.hnl"
    code, rec, _err = _run(capsys, monkeypatch, "deliver", _deliver_input(verify, out))
    assert code == 1
    assert rec["ok"] is False
    assert rec["delivered"] is False
    assert "arity mismatch" in rec["error"]
    assert not out.exists()


def test_record_source_drift_covers_inputs_and_params() -> None:
    """F10: the re-export drift check compares the WHOLE lifted IR —
    root term, declared inputs and the param table — not just the
    root (leaves are name/shape-keyed terms)."""
    from mlopt_core.ir import IR, Op, Param, TensorType, Var

    from mlopt.stage import _RecordSource

    x = Var("x", TensorType((2,)))
    y = Var("y", TensorType((1,)))
    w = Param("w", TensorType((2,)))
    root = Op.make("add", x, w)
    ir = IR(root=root, inputs=[x], input_names={"x"}, params={"w": w})

    class _Inner:
        def __init__(self, fresh):  # noqa: ANN001, ANN202
            self._fresh = fresh

        def to_ir(self, model, example_inputs):  # noqa: ANN001, ANN202
            return self._fresh, {}

    # Identical re-export → no drift, and the RECORD's IR is served.
    src = _RecordSource(_Inner(ir), ir)
    out_ir, _leaves = src.to_ir(None, None)
    assert out_ir is ir
    assert src.drifted is False

    # Same root, extra declared input → drift.
    fresh_inputs = IR(root=root, inputs=[x, y], input_names={"x", "y"}, params={"w": w})
    src = _RecordSource(_Inner(fresh_inputs), ir)
    src.to_ir(None, None)
    assert src.drifted is True

    # Same root + inputs, param table disagrees → drift.
    w2 = Param("w2", TensorType((2,)))
    fresh_params = IR(
        root=root,
        inputs=[x],
        input_names={"x"},
        params={"w": w, "w2": w2},
    )
    src = _RecordSource(_Inner(fresh_params), ir)
    src.to_ir(None, None)
    assert src.drifted is True


# --- protocol edges ------------------------------------------------------------


def test_bad_stdin_json_exits_2(capsys, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("sys.stdin", io.StringIO("this is not json"))
    code = main(["stage", "lift"])
    captured = capsys.readouterr()
    assert code == 2
    assert "not one JSON value" in captured.err


def test_unknown_stage_rejected_by_parser(capsys, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr("sys.stdin", io.StringIO("null"))
    with pytest.raises(SystemExit):
        main(["stage", "explode"])


def test_subprocess_protocol_roundtrip(tiny_hnl) -> None:
    """The real ``python -m mlopt stage`` channel: one JSON value each way."""
    inp = json.dumps(_lift_input(tiny_hnl))
    lift = subprocess.run(
        [sys.executable, "-m", "mlopt", "stage", "lift"],
        input=inp,
        capture_output=True,
        text=True,
        check=True,
    )
    rec = json.loads(lift.stdout)  # exactly one JSON value on stdout
    assert rec["ok"] is True and rec["stage"] == "lift"

    for stage in ("search", "verify"):
        out = subprocess.run(
            [sys.executable, "-m", "mlopt", "stage", stage],
            input=json.dumps(rec),
            capture_output=True,
            text=True,
            check=True,
        )
        rec = json.loads(out.stdout)
        assert rec["ok"] is True and rec["stage"] == stage
    assert rec["verified"]["passed"] is True


# --- record schema version (ADR-0178 WP0) ------------------------------------


def test_records_carry_schema_version(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    """Every emitted record pins ``"v": RECORD_VERSION`` — including the
    ``ok:false`` declines, which are records too."""
    from mlopt.stage import RECORD_VERSION

    lift, search, verify, _ = _pipeline(capsys, monkeypatch, tiny_hnl)
    for rec in (lift, search, verify):
        assert rec["v"] == RECORD_VERSION == 3

    # A decline is still a record — it carries the pin as well.
    code, rec, _ = _run(capsys, monkeypatch, "search", {"stage": "lift"})
    assert code == 1 and rec["ok"] is False
    assert rec["v"] == RECORD_VERSION


def test_unversioned_record_reads_as_legacy_v0(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    """A v0 record (no ``v`` — the pre-versioning shape) is accepted and
    re-emitted at the current pin, so old pipelines keep flowing."""
    code, lift, _ = _run(capsys, monkeypatch, "lift", _lift_input(tiny_hnl))
    assert code == 0 and lift["ok"] is True

    # Strip the pin — this is what a pre-versioning record looks like.
    v0 = {k: v for k, v in lift.items() if k != "v"}
    assert "v" not in v0

    code, search, _ = _run(capsys, monkeypatch, "search", v0)
    assert code == 0 and search["ok"] is True
    assert search["v"] == 3

    code, verify, _ = _run(capsys, monkeypatch, "verify", search)
    assert code == 0 and verify["ok"] is True
    assert verify["v"] == 3


def test_future_record_version_declines(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    """A ``v`` newer than this endpoint's schema is a named decline —
    never silently misread."""
    _, lift, _ = _run(capsys, monkeypatch, "lift", _lift_input(tiny_hnl))

    future = dict(lift, v=99)
    code, rec, err = _run(capsys, monkeypatch, "search", future)
    assert code == 1 and rec["ok"] is False
    assert "newer than this endpoint" in rec["error"]

    code, rec, err = _run(capsys, monkeypatch, "lift", dict(future, model=str(tiny_hnl)))
    assert code == 1 and rec["ok"] is False
    assert "newer than this endpoint" in rec["error"]


def test_malformed_version_declines(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    """Non-int ``v`` (including bools) is malformed, not versioned."""
    _, lift, _ = _run(capsys, monkeypatch, "lift", _lift_input(tiny_hnl))
    for bad in ("1", 1.0, True, [1]):
        code, rec, _ = _run(capsys, monkeypatch, "search", dict(lift, v=bad))
        assert code == 1 and rec["ok"] is False, bad
        assert "'v'" in rec["error"]


# --- Term-JSON codec (eqsat v0.1.12 — mlopt_core.term_json) --------------------


def test_v3_payloads_emit_canonical_term_json(tiny_hnl, capsys, monkeypatch) -> None:  # noqa: ANN001
    """v3 records carry the canonical ``mlopt_core.term_json`` spelling:
    ``Arr`` leaves, the ``attrs`` channel, ``source: "Native"``.

    ``const-str`` leaf payloads must NEVER appear in stage records —
    pre-shim ``hnl_codec`` readers misdecode them as scalar ``Const``
    values, silently losing ``Var``/``Param`` typing.
    """
    code, lift, _ = _run(capsys, monkeypatch, "lift", _lift_input(tiny_hnl))
    assert code == 0
    blob = json.dumps(lift["ir"])
    # Canonical structured-leaf spelling is present...
    assert '"__mlopt_leaf__"' in blob
    # ...and the forbidden spellings are absent: const-str leaf payloads
    # (``"__mlopt_leaf__:{...}"`` string Consts), Foreign sources and the
    # legacy ``lit.__mlopt_attrs__`` attrs channel.
    assert "__mlopt_leaf__:" not in blob
    assert '"Foreign"' not in blob
    assert "__mlopt_attrs__" not in blob
    root = lift["ir"]["root"]["Op"]
    assert root["source"] == "Native"


def test_term_decode_accepts_legacy_hnl_codec_v3_payload() -> None:
    """Old v3 records — spelled by the pre-shim
    ``kit_discovery.hnl_codec`` (``Foreign`` source, ``lit`` attrs
    channel, nested-record dag ``dim``, ``__mlopt_leaf__:``-prefixed
    ``Const``-string leaves) — still decode through ``_term_decode``:
    decode is permanently tolerant of every legacy spelling.

    The payload is hand-built: ``kit_discovery.hnl_codec`` ≥ v0.1.4 is
    a ``mlopt_core.term_json`` shim and no longer emits these
    spellings, so the legacy record must be constructed literally."""
    from mlopt_core.term import Op, Param, TensorType, Var

    from mlopt.stage import _term_decode

    x = Var("x", TensorType((2,)))
    w = Param("w", TensorType((2,)))
    sub = Op.make("mul", x, w)
    term = Op.make("add", sub, sub)

    def leaf(kind: str, name: str) -> dict:
        # const-str leaf: "__mlopt_leaf__:" + {"k","n","s"} JSON
        return {
            "Const": {
                "value": "__mlopt_leaf__:"
                + json.dumps({"k": kind, "n": name, "s": [2]}, separators=(",", ":"))
            }
        }

    foreign = {"Foreign": "tensor"}
    payload = {
        "Opaque": {
            "__tag__": "dag.program",
            "nodes": [
                leaf("var", "x"),
                leaf("param", "w"),
                {
                    "Op": {
                        "name": "mul",
                        "args": [0, 1],
                        # nested-record dim — the legacy dag spelling
                        "dim": {"Const": {"value": 1}},
                        "source": foreign,
                    }
                },
                {
                    "Op": {
                        "name": "add",
                        "args": [2, 2],
                        "dim": {"Const": {"value": 1}},
                        "source": foreign,
                    }
                },
            ],
            "root": 3,
        }
    }
    assert '"Foreign"' in json.dumps(payload)
    back = _term_decode(payload, "ir.root")
    assert back == term


def test_term_decode_accepts_const_str_leaf_payload() -> None:
    """``const-str`` leaves (the eqsat engine-wire profile) decode in
    stage records too — decode accepts both leaf spellings forever,
    even though emit never produces them."""
    from mlopt_core.term import TensorType, Var

    from mlopt.stage import _term_decode

    leaf = {"Const": {"value": "__mlopt_leaf__:" + json.dumps({"k": "var", "n": "x", "s": [2]})}}
    payload = {
        "Op": {
            "name": "mul",
            "args": [leaf, {"Const": {"value": 1.0}}],
            "dim": {"Const": {"value": 1}},
            "source": "Native",
        }
    }
    term = _term_decode(payload, "t")
    leaf_out = term.args[0]
    assert isinstance(leaf_out, Var)
    assert leaf_out.name == "x"
    assert leaf_out.typ == TensorType((2,))
