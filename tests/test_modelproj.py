"""Phase-2 Models: spec→``.hnl`` projection, program emit, fp64 gate.

``modelproj`` is the spec-side bridge ``term_hnl`` is for terms:
denotation specs and ``ew`` handler bodies become ``tensor.*``
expressions, ``hnl_program`` binds fp64 feeds the way
``emit_program`` does, and ``verify_model`` is the fp64 gate —
``handl-shipped`` only when every bound site holds the shipped
``allclose`` floor.  The eval callable is injected, so the whole gate
is exercised without the ``hnl-run`` binary (the e2e parity test in
``test_vocab_model_parity.py`` runs the real one).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from mlopt_discovery import modelproj, vocabdir

_REPO = Path(__file__).resolve().parents[1]
_VOCAB = _REPO / "vocab"


# ---------------------------------------------------------------------------
#  spec_mvars / body_mvars
# ---------------------------------------------------------------------------


def test_spec_mvars_skips_op_heads_and_attrs() -> None:
    spec = ("add", ("mul", "x", "y"), ("mul", "z", 2), {"note": "s"})
    # the trailing dict is attrs — its strings are not metavars
    assert modelproj.spec_mvars(("add", "x", "y")) == {"x", "y"}
    assert modelproj.spec_mvars(spec) == {"x", "y", "z"}


def test_body_mvars_reads_the_ew_payload() -> None:
    body = ("ew", ("add", "a", ("mul", "b", "c")))
    assert modelproj.body_mvars(body) == {"a", "b", "c"}
    assert modelproj.body_mvars("not-a-body") == set()
    assert modelproj.body_mvars(("ew",)) == set()


# ---------------------------------------------------------------------------
#  spec_to_hnl / handler_body_to_hnl
# ---------------------------------------------------------------------------


def test_spec_to_hnl_elementwise() -> None:
    hnl = modelproj.spec_to_hnl(("add", ("mul", "x", "y"), "z"))
    assert hnl == "tensor.add(tensor.mul(x, y), z)"


def test_spec_to_hnl_all_binary_ops() -> None:
    for op in ("add", "sub", "mul", "div", "matmul"):
        assert modelproj.spec_to_hnl((op, "a", "b")) == (
            f"tensor.{op}(a, b)"
        )


def test_spec_to_hnl_consts_and_attrs() -> None:
    hnl = modelproj.spec_to_hnl(
        ("reshape", "x", {"shape": [2, 3]}), mvars={"x"}
    )
    assert hnl == "tensor.reshape(x, [2, 3])"
    hnl = modelproj.spec_to_hnl(
        ("transpose", "x", {"dim0": 0, "dim1": 2}), mvars={"x"}
    )
    assert hnl == "tensor.transpose(x, 0, 2)"
    hnl = modelproj.spec_to_hnl(
        ("concat", "a", "b", {"dim": 1}), mvars={"a", "b"}
    )
    assert hnl == "tensor.concat([a, b], 1)"
    # scalar const leaves render through _num (no scientific notation)
    assert (
        modelproj.spec_to_hnl(("mul", "x", 0.00001), mvars={"x"})
        == "tensor.mul(x, 0.00001)"
    )


def test_spec_to_hnl_sanitizes_metavar_names() -> None:
    hnl = modelproj.spec_to_hnl(
        ("add", "a.b", "tensor"), mvars={"a.b", "tensor"}
    )
    # "a.b" → "a_b"; the literal name "tensor" is reserved → "v_tensor"
    assert hnl == "tensor.add(a_b, v_tensor)"


def test_spec_to_hnl_refuses_unknown_ops_by_name() -> None:
    # ``unsqueeze`` joined the surface in phase 3 (host-tensor
    # ``tensor.unsqueeze``); ``select`` remains off it.
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="select"
    ):
        modelproj.spec_to_hnl(("select", "x", {"dim": 0, "index": 0}))
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="frobnicate"
    ):
        modelproj.spec_to_hnl(("frobnicate", "x"))


def test_spec_to_hnl_refuses_bad_arity_and_attrs() -> None:
    with pytest.raises(modelproj.ModelUnsupportedError, match="arity"):
        modelproj.spec_to_hnl(("add", "x"))
    with pytest.raises(modelproj.ModelUnsupportedError, match="attrs"):
        modelproj.spec_to_hnl(("add", "x", "y", {"dim": 0}))
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="reshape"
    ):
        modelproj.spec_to_hnl(
            ("reshape", "x", {"shape": [2], "extra": 1})
        )


def test_handler_body_to_hnl_ew_only() -> None:
    body = ("ew", ("add", "a", ("mul", "b", "c")))
    assert modelproj.handler_body_to_hnl(body) == (
        "tensor.add(a, tensor.mul(b, c))"
    )
    for tag in (
        "gather",
        "family",
        "contract",
        "fold",
        "reduce",
        "norm",
    ):
        with pytest.raises(modelproj.ModelUnsupportedError, match=tag):
            modelproj.handler_body_to_hnl((tag, "sub", "x"))
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="malformed"
    ):
        modelproj.handler_body_to_hnl("junk")
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="malformed"
    ):
        modelproj.handler_body_to_hnl(())


# ---------------------------------------------------------------------------
#  tensor_feed / hnl_program / decode_tensor_result / max_rel
# ---------------------------------------------------------------------------


def test_tensor_feed_row_major_fp64() -> None:
    t = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    f = modelproj.tensor_feed(t)
    assert f == {
        "shape": [2, 3],
        "data": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    }
    with pytest.raises(modelproj.ModelUnsupportedError, match="tensor"):
        modelproj.tensor_feed([1.0, 2.0])


def test_hnl_program_binds_feeds_in_sorted_order() -> None:
    feed = {
        "b": {"shape": [2], "data": [3.0, 4.0]},
        "a": {"shape": [2], "data": [1.0, 2.0]},
    }
    src = modelproj.hnl_program(
        "tensor.add(a, b)", feed, mvars={"a", "b"}
    )
    assert src.startswith("do {")
    assert src.endswith("}")
    assert "  a :: tensor.reshape" in src
    assert "  b :: tensor.reshape" in src
    # 'a' binds before 'b' (sorted), and the expr is the last line
    assert (
        src.index("a ::") < src.index("b ::") < src.rindex("tensor.add")
    )


def test_decode_tensor_result() -> None:
    shape, data = modelproj.decode_tensor_result(
        {"__tensor__": {"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0]}}
    )
    assert shape == [2, 2] and data == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(modelproj.ModelUnsupportedError, match="tensor"):
        modelproj.decode_tensor_result([1.0, 2.0])
    with pytest.raises(modelproj.ModelUnsupportedError, match="tensor"):
        modelproj.decode_tensor_result({"other": 1})


def test_max_rel_normalizes_by_ref_plus_atol() -> None:
    assert modelproj.max_rel([1.0, 2.0], [1.0, 2.0], atol=1e-12) == 0.0
    # |3-2| / (|2| + atol) ≈ 0.5
    assert modelproj.max_rel([3.0], [2.0], atol=1e-12) == pytest.approx(
        0.5
    )
    # ref = 0 → denominator is atol alone
    assert modelproj.max_rel(
        [1e-13], [0.0], atol=1e-12
    ) == pytest.approx(1e-13 / 1e-12)
    with pytest.raises(ValueError, match="mismatched"):
        modelproj.max_rel([1.0], [1.0, 2.0], atol=1e-12)


# ---------------------------------------------------------------------------
#  verify_model — the fp64 gate, with an injected eval
# ---------------------------------------------------------------------------


def _verify(
    eval_hnl,
    *,
    ref_spec=(("add", "x", "y")),
    domain=({"x": [2, 3], "y": [2, 3]},),
    hnl="tensor.add(x, y)",
    adversarial=0,
):
    return modelproj.verify_model(
        name="test_op",
        hnl=hnl,
        ref_spec=ref_spec,
        domain=list(domain),
        mvars=["x", "y"],
        eval_hnl=eval_hnl,
        adversarial=adversarial,
    )


def test_verify_model_handl_shipped_on_exact_eval() -> None:
    """The positive gate path: every draw allclose → ``handl-shipped``.

    ``eval_hnl`` here replays the reference tensors ``_domain_draws``
    will produce (same seed, same order) — it exercises the gate's
    draw/decode/compare/verdict logic end to end; the real ``hnl-run``
    arithmetic parity is ``test_vocab_model_parity.py``'s job (e2e).
    """
    import mlopt_torch.torch_bridge as tb
    from mlopt_discovery import handler_synth

    ref_spec = ("add", "x", "y")
    domain = [{"x": [2, 3], "y": [2, 3]}]
    _envs, refs = handler_synth._domain_draws(
        ref_spec, domain, 0, tb._IR_TO_TORCH, 0
    )
    queue = [
        {
            "__tensor__": {
                "shape": [int(d) for d in ref.shape],
                "data": [float(v) for v in ref.reshape(-1).tolist()],
            }
        }
        for ref in refs
    ]

    def eval_hnl(src: str):
        assert "tensor.add" in src
        return queue.pop(0)

    v = modelproj.verify_model(
        name="test_op",
        hnl="tensor.add(x, y)",
        ref_spec=ref_spec,
        domain=domain,
        mvars=["x", "y"],
        eval_hnl=eval_hnl,
        adversarial=0,
        bindings=tb._IR_TO_TORCH,
    )
    assert v["status"] == "handl-shipped"
    assert v["sites"] == len(refs)
    assert v["max_err"] == 0.0
    assert v["max_rel"] == 0.0
    assert queue == []


def test_verify_model_refusal_paths() -> None:
    def raises(src: str):
        raise RuntimeError("kit exploded")

    v = _verify(raises)
    assert v["status"] == "python-shipped"
    assert "kit exploded" in v["detail"]

    def wrong_shape(src: str):
        return {"__tensor__": {"shape": [3, 2], "data": [0.0] * 6}}

    v = _verify(wrong_shape)
    assert v["status"] == "python-shipped"
    assert "shape mismatch" in v["detail"]

    def not_tensor(src: str):
        return [1.0, 2.0]

    v = _verify(not_tensor)
    assert v["status"] == "python-shipped"
    assert "hnl eval failed" in v["detail"]


def test_verify_model_undrawable_domain() -> None:
    # a domain whose shapes cannot broadcast → DomainShapeError path
    v = modelproj.verify_model(
        name="t",
        hnl="tensor.add(x, y)",
        ref_spec=("add", "x", "y"),
        domain=[{"x": [2, 3], "y": [4, 5]}],
        mvars=["x", "y"],
        eval_hnl=lambda s: {},
        adversarial=0,
    )
    assert v["status"] == "python-shipped"
    assert "no drawable domain" in v["detail"]


def test_verify_model_verdict_shape() -> None:
    """Every verdict carries the provenance keys the gate records."""
    v = _verify(
        lambda s: {"__tensor__": {"shape": [2, 3], "data": [0.0] * 6}}
    )
    assert set(v) >= {
        "op",
        "status",
        "rtol",
        "atol",
        "max_err",
        "max_rel",
        "sites",
        "detail",
    }


# ---------------------------------------------------------------------------
#  vocabdir integration — models.yaml is canonical
# ---------------------------------------------------------------------------


def test_lit_and_num_edge_cases() -> None:
    """Literal rendering: bools, None, strings, lists, refusals."""
    assert modelproj._num(True) == "true"
    assert modelproj._num(False) == "false"
    assert modelproj._num(3) == "3"
    assert modelproj._num(0.5) == "0.5"
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="non-finite"
    ):
        modelproj._num(float("nan"))
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="non-finite"
    ):
        modelproj._num(float("inf"))
    assert modelproj._lit(None) == "null"
    assert modelproj._lit("s") == '"s"'
    assert modelproj._lit([1, "a"]) == '[1, "a"]'
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="unemittable"
    ):
        modelproj._lit(object())


def test_sanitize_collisions_and_model_names() -> None:
    names = modelproj.model_names({"a-b", "a_b", "tensor", "9lives"})
    # "a-b" and "a_b" both sanitize to "a_b" — sorted order decides
    vals = sorted(names.values())
    assert len(set(vals)) == len(vals)
    assert "v_tensor" in vals and "v_9lives" in vals


def test_spec_mvars_nonlinear_structures() -> None:
    # a tuple headed by a string is a call — the head is an op, not a
    # metavar; a non-str-headed tuple walks every element
    assert modelproj.spec_mvars((("add", "x", "y"), "z")) == {
        "x",
        "y",
        "z",
    }
    assert modelproj.spec_mvars(((1, "q"), "z")) == {"q", "z"}
    # non-str scalars contribute nothing
    assert modelproj.spec_mvars(("add", "x", 2, None, True)) == {"x"}


def test_spec_to_hnl_data_list_leaf() -> None:
    # a non-call list is data — emitted as a literal list
    hnl = modelproj.spec_to_hnl([1, 2, 3])
    assert hnl == "[1, 2, 3]"
    # scalar leaves reach _lit through the fallback
    assert modelproj.spec_to_hnl(None) == "null"
    assert modelproj.spec_to_hnl(True) == "true"


def test_emit_spec_refusals() -> None:
    names = {"x": "x"}
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="missing"
    ):
        modelproj._emit_spec("ghost", names)
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="call position"
    ):
        modelproj._emit_spec({"shape": [2]}, names)
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="unemittable"
    ):
        modelproj._emit_spec(object(), names)


def test_concat_and_attr_op_refusals() -> None:
    with pytest.raises(modelproj.ModelUnsupportedError, match="concat"):
        modelproj.spec_to_hnl(("concat", "a", "b", {"dim": 0, "x": 1}))
    with pytest.raises(modelproj.ModelUnsupportedError, match="arity"):
        modelproj.spec_to_hnl(("reshape", "x", "y", {"shape": [2]}))
    with pytest.raises(
        modelproj.ModelUnsupportedError, match="transpose"
    ):
        modelproj.spec_to_hnl(("transpose", "x", {"dim0": 0}))


def test_hnl_program_feed_variants() -> None:
    src = modelproj.hnl_program(
        "tensor.add(a, b)",
        {
            "a": {"hnl": "tensor.ones([2])"},
            "b": [1.0, 2.0],
        },
        mvars={"a", "b"},
    )
    assert "a :: tensor.ones([2])" in src
    assert "b :: tensor.from_list([1.0, 2.0])" in src


def test_verify_model_eval_and_decode_failures() -> None:
    # decode_tensor_result raising inside eval_hnl surfaces as eval fail
    def bad_decode(src: str):
        return {"not": "a tensor"}

    v = _verify(bad_decode)
    assert v["status"] == "python-shipped"
    assert "hnl eval failed" in v["detail"]


def test_models_yaml_is_canonical_and_fresh() -> None:
    """The shipped vocab carries models.yaml and passes --check."""
    assert "models.yaml" in vocabdir.CANONICAL_FILES
    assert (_VOCAB / "models.yaml").is_file()
    assert vocabdir.check_vocab(_VOCAB) == []


def test_load_vocab_exposes_models() -> None:
    v = vocabdir.load_vocab(_VOCAB)
    assert len(v.models) == 109
    kinds = {r["kind"] for r in v.models}
    assert kinds == {"denotation", "handler"}
    expressible = [r for r in v.models if r.get("hnl")]
    # phase 3: ``tensor.unsqueeze`` joined the host surface, so the
    # four unsqueeze-blocked denotations now project; only the
    # ``select``-carrying denotation stays unexpressible.
    assert len(expressible) == 108
    unsupported = [r for r in v.models if r.get("unsupported")]
    assert len(unsupported) == 1
    assert all(r["hnl"] is None for r in unsupported)
    # every expressible model names its op and mvars
    for r in expressible:
        assert r["op"].startswith("synop")
        assert r["mvars"]


def test_collect_models_matches_shipped_prims() -> None:
    """57 denotation rows + 52 handler rows = 109."""
    doc = vocabdir._COLLECTORS["models.yaml"]()
    rows = doc["models"]
    assert len(rows) == 109
    assert sum(1 for r in rows if r["kind"] == "denotation") == 57
    assert sum(1 for r in rows if r["kind"] == "handler") == 52


def test_emit_all_tmpdir_models_and_verdict_roundtrip(
    tmp_path: Path,
) -> None:
    """Emit to a scratch dir; verdicts upsert and survive re-emit."""
    vocabdir.emit_all(tmp_path)
    v = vocabdir.load_vocab(tmp_path)
    assert len(v.models) == 109
    assert v.model_verdicts == {}

    vocabdir.upsert_model_verdicts(
        tmp_path,
        {
            "denotation:synop2_35bbd03301": {
                "op": "synop2_35bbd03301",
                "status": "handl-shipped",
                "max_rel": 0.0,
                "hnl": "tensor.add(x, y)",
            }
        },
    )
    v = vocabdir.load_vocab(tmp_path, fresh=True)
    assert (
        v.model_verdicts["denotation:synop2_35bbd03301"]["status"]
        == "handl-shipped"
    )
    # re-emission preserves the verdict (verdicts are not owner rows)
    vocabdir.emit_all(tmp_path)
    v = vocabdir.load_vocab(tmp_path, fresh=True)
    assert "denotation:synop2_35bbd03301" in v.model_verdicts
    assert vocabdir.check_vocab(tmp_path) == []


def test_collect_models_handler_unsupported_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A minted handler whose body family has no tensor.* spelling is
    recorded ``unsupported`` — the honest-skip row, not a crash."""
    from mlopt_core.laws import minted

    fake = {
        "synop_fake": {
            "handler": {
                "family": "gather",
                "record": {
                    "kind": "handler",
                    "pattern": ["x"],
                    "kernel": "synop_fake",
                    "args": ["x"],
                    "handler_body": ["gather", ["x"]],
                },
            }
        }
    }
    monkeypatch.setattr(minted, "prim_records", lambda: fake)
    doc = vocabdir._COLLECTORS["models.yaml"]()
    row = next(
        r for r in doc["models"] if r["id"] == "handler:synop_fake"
    )
    assert row["hnl"] is None
    assert "gather" in row["unsupported"]
    assert row["family"] == "gather"


def test_check_vocab_flags_edited_model_row(tmp_path: Path) -> None:
    vocabdir.emit_all(tmp_path)
    path = tmp_path / "models.yaml"
    doc = json.loads(path.read_text())
    doc["models"][0]["hnl"] = "tensor.mul(x, y)  # hand-edit"
    path.write_text(json.dumps(doc))
    drift = vocabdir.check_vocab(tmp_path)
    assert any("models.yaml" in d for d in drift)
