"""``mlopt_discovery.hnl_codec`` — the Term-JSON wire codec (ADR-0173).

Pins the codec-envelope half of ADR-0173: the ``__nonfinite__``
tagged-object spelling for NaN/±inf wherever a domain scalar lands in
a ``Value`` position (``Const`` values, attr payloads — nested
included), and the ``dag.program`` node-table record that carries DAG
sharing inside ``Term::Opaque`` — plus the honest declines (shared
term under ``dag=False``, the record under ``allow_dag=False``,
unknown ``Opaque`` tags, malformed node tables).
"""

from __future__ import annotations

import json
import math

import pytest
from mlopt_core.ir import Const, Op, Param, TensorType, Var
from mlopt_discovery.hnl_codec import (
    HnlCodecError,
    term_from_json,
    term_to_json,
)


def _x() -> Var:
    return Var("x", TensorType((2, 3)))


def _w() -> Param:
    return Param("w", TensorType((3, 4)))


def _assert_nan(v: float) -> None:
    assert isinstance(v, float) and math.isnan(v)


# ---------------------------------------------------------------------------
#  Round-trip — the tree envelope
# ---------------------------------------------------------------------------


class TestTreeRoundTrip:
    def test_unshared_term_is_plain_tree(self) -> None:
        t = Op.make("add", _x(), _w())
        j = term_to_json(t)
        assert "Op" in j and "Opaque" not in j
        assert term_from_json(j) == t

    def test_leaf_kinds_round_trip(self) -> None:
        for leaf in (_x(), _w(), Const(3), Const(2.5)):
            assert term_from_json(term_to_json(leaf)) == leaf

    def test_attrs_round_trip(self) -> None:
        t = Op.make(
            "conv2d",
            _x(),
            _w(),
            stride=[1, 1],
            padding=[0, 0],
            groups=1,
        )
        assert term_from_json(term_to_json(t)) == t

    def test_json_safe_output(self) -> None:
        t = Op.make("clamp", _x(), min=-1.0, max=1.0, validate=False)
        # allow_nan=False proves no bare NaN/Infinity literal escaped.
        blob = json.dumps(term_to_json(t), allow_nan=False)
        assert term_from_json(json.loads(blob)) == t

    def test_unsupported_leaf_declines(self) -> None:
        with pytest.raises(TypeError, match="unencodable"):
            term_to_json("m")  # a metavar string is not a program


# ---------------------------------------------------------------------------
#  ``__nonfinite__`` — the codec tag
# ---------------------------------------------------------------------------


class TestNonfinite:
    @pytest.mark.parametrize(
        "value,check",
        [
            (float("nan"), _assert_nan),
            (float("inf"), lambda v: v == float("inf")),
            (float("-inf"), lambda v: v == float("-inf")),
        ],
    )
    def test_const_nonfinite_round_trip(self, value, check) -> None:
        t = Const(value)
        j = term_to_json(t)
        # the wire spelling is the tagged object, never a bare literal
        wire = j["Const"]["value"]
        assert isinstance(wire, dict) and set(wire) == {"__nonfinite__"}
        json.dumps(j, allow_nan=False)  # would raise on a bare NaN
        back = term_from_json(j)
        assert isinstance(back, Const)
        check(back.value)

    def test_nonfinite_in_attrs(self) -> None:
        t = Op.make(
            "clamp",
            _x(),
            min=float("nan"),
            max=float("inf"),
            validate=False,
        )
        j = term_to_json(t)
        attrs = j["Op"]["lit"]["__mlopt_attrs__"]
        assert attrs["min"] == {"__nonfinite__": "nan"}
        assert attrs["max"] == {"__nonfinite__": "inf"}
        json.dumps(j, allow_nan=False)
        back = term_from_json(j)
        assert back == t
        _assert_nan(back.attrs["min"])
        assert back.attrs["max"] == float("inf")

    def test_nonfinite_nested_in_attr_containers(self) -> None:
        t = Op.make(
            "pad",
            _x(),
            pad=(float("-inf"), 1, 2),
            validate=False,
        )
        j = term_to_json(t)
        wire = j["Op"]["lit"]["__mlopt_attrs__"]["pad"]
        # attr_to_data spells the tuple as {"__tuple__": [...]} — the
        # tag nests inside, and decode restores both layers.
        assert wire["__tuple__"][0] == {"__nonfinite__": "-inf"}
        back = term_from_json(j)
        assert back == t
        assert back.attrs["pad"][0] == float("-inf")

    def test_decode_handwritten_tag(self) -> None:
        j = {"Const": {"value": {"__nonfinite__": "-inf"}}}
        back = term_from_json(j)
        assert isinstance(back, Const)
        assert back.value == float("-inf")

    def test_bad_tag_payload_declines(self) -> None:
        j = {"Const": {"value": {"__nonfinite__": "bogus"}}}
        with pytest.raises(HnlCodecError, match="__nonfinite__"):
            term_from_json(j)

    def test_finite_floats_untouched(self) -> None:
        t = Op.make("clamp", _x(), min=0.5, max=-2.25, validate=False)
        j = term_to_json(t)
        attrs = j["Op"]["lit"]["__mlopt_attrs__"]
        assert attrs == {"min": 0.5, "max": -2.25}
        assert term_from_json(j) == t

    def test_nonfinite_inside_dag_record(self) -> None:
        nan = Const(float("nan"))
        shared = Op.make("add", _x(), nan, validate=False)
        root = Op.make("mul", shared, shared)
        j = term_to_json(root)
        rec = j["Opaque"]
        consts = [n for n in rec["nodes"] if "Const" in n]
        assert consts[0]["Const"]["value"] == {"__nonfinite__": "nan"}
        back = term_from_json(j)
        assert back == root
        _assert_nan(back.args[0].args[1].value)


# ---------------------------------------------------------------------------
#  ``dag.program`` — the sharing envelope
# ---------------------------------------------------------------------------


def _shared_term() -> Op:
    """A small DAG: ``s1`` feeds three parents, ``s2`` two."""
    s1 = Op.make("add", _x(), _w())
    s2 = Op.make("mul", s1, s1)  # s1 twice under one parent
    s3 = Op.make("sub", s2, s1)  # s1 again — three parents total
    return Op.make("mul", s3, s2)


class TestDagRecord:
    def test_shared_term_encodes_as_record(self) -> None:
        t = _shared_term()
        j = term_to_json(t)
        assert set(j) == {"Opaque"}
        rec = j["Opaque"]
        assert rec["__tag__"] == "dag.program"
        assert isinstance(rec["nodes"], list)
        assert isinstance(rec["root"], int)
        # compactness: 6 distinct nodes vs the tree's 9 spellings
        assert len(rec["nodes"]) == 6
        # Op args are node-table indices, children before parents
        for i, n in enumerate(rec["nodes"]):
            if "Op" in n:
                for a in n["Op"]["args"]:
                    assert isinstance(a, int) and 0 <= a < i

    def test_dag_round_trip_preserves_sharing(self) -> None:
        t = _shared_term()
        j = term_to_json(t)
        back = term_from_json(j)
        assert back == t
        # sharing is reconstructed, not re-expanded: count child
        # occurrences over the decoded DAG — shared nodes keep >1
        # parent instead of being duplicated per use site.
        seen: dict[int, int] = {}

        def count(n: object) -> None:
            if isinstance(n, Op):
                for a in n.args:
                    seen[id(a)] = seen.get(id(a), 0) + 1
                    count(a)

        count(back)
        shared_ids = {i for i, c in seen.items() if c > 1}
        assert len(shared_ids) >= 2  # s1 (x3) and s2 (x2)
        # re-encoding the decode is a fixed point — same record
        assert term_to_json(back) == j

    def test_unshared_term_stays_tree_under_auto(self) -> None:
        t = Op.make("add", Op.make("mul", _x(), _w()), Const(1))
        j = term_to_json(t)
        assert "Op" in j

    def test_dag_true_forces_record(self) -> None:
        t = Op.make("add", _x(), _w())
        j = term_to_json(t, dag=True)
        assert j["Opaque"]["__tag__"] == "dag.program"
        assert term_from_json(j) == t

    def test_dag_false_declines_shared_term(self) -> None:
        with pytest.raises(HnlCodecError, match="shared subtrees"):
            term_to_json(_shared_term(), dag=False)

    def test_dag_false_allows_unshared_term(self) -> None:
        t = Op.make("add", _x(), _w())
        assert term_from_json(term_to_json(t, dag=False)) == t

    def test_allow_dag_false_declines_record(self) -> None:
        j = term_to_json(_shared_term())
        with pytest.raises(HnlCodecError, match="cannot represent"):
            term_from_json(j, allow_dag=False)

    def test_unknown_opaque_tag_declines(self) -> None:
        j = {"Opaque": {"__tag__": "something.else", "data": 1}}
        with pytest.raises(HnlCodecError, match="Opaque"):
            term_from_json(j)

    def test_untyped_opaque_declines(self) -> None:
        with pytest.raises(HnlCodecError, match="Opaque"):
            term_from_json({"Opaque": {"raw": True}})

    def test_non_dict_encoding_declines(self) -> None:
        with pytest.raises(HnlCodecError, match="bad term encoding"):
            term_from_json(["not", "a", "dict"])

    def test_unknown_node_kind_declines(self) -> None:
        with pytest.raises(HnlCodecError, match="bad leaf encoding"):
            term_from_json({"mvar": "x"})

    def test_unencodable_leaf_in_dag_declines(self) -> None:
        # a metavar string leaf shared twice routes through the dag
        # encoder, which refuses it exactly like the tree encoder
        t = Op.make("add", "m", "m", validate=False)
        with pytest.raises(TypeError, match="unencodable"):
            term_to_json(t)
        with pytest.raises(TypeError, match="unencodable"):
            term_to_json(t, dag=True)

    def test_unserializable_const_declines(self) -> None:
        with pytest.raises(TypeError, match="unserializable"):
            term_to_json(Const({1, 2}))  # type: ignore[arg-type]

    def test_dag_node_args_not_list_declines(self) -> None:
        j = {
            "Opaque": {
                "__tag__": "dag.program",
                "nodes": [{"Op": {"name": "neg"}}],
                "root": 0,
            }
        }
        with pytest.raises(HnlCodecError, match="index list"):
            term_from_json(j)

    def test_leaf_only_dag_round_trip(self) -> None:
        j = term_to_json(_w(), dag=True)
        assert j["Opaque"]["__tag__"] == "dag.program"
        assert term_from_json(j) == _w()

    def test_dag_node_with_attrs_round_trip(self) -> None:
        # an attr-carrying Op inside the node table keeps its
        # __mlopt_attrs__ lit record
        s = Op.make("clamp", _x(), min=0.25, validate=False)
        t = Op.make("mul", s, s)
        j = term_to_json(t)
        ops = [n["Op"] for n in j["Opaque"]["nodes"] if "Op" in n]
        assert any("lit" in o for o in ops)
        assert term_from_json(j) == t

    def test_nested_non_dict_arg_declines(self) -> None:
        j = {"Op": {"name": "neg", "args": ["x"]}}
        with pytest.raises(HnlCodecError, match="bad term encoding"):
            term_from_json(j)

    def test_nested_opaque_declines(self) -> None:
        inner = {
            "Opaque": {"__tag__": "dag.program", "nodes": [], "root": 0}
        }
        j = {
            "Op": {
                "name": "add",
                "args": [inner, inner],
                "source": {"Foreign": "tensor"},
            }
        }
        with pytest.raises(HnlCodecError, match="nested Opaque"):
            term_from_json(j)


class TestDagRecordValidation:
    """Malformed node tables decline loudly, never guess."""

    def _rec(self, **kw):
        rec = {"__tag__": "dag.program", "nodes": [], "root": 0}
        rec.update(kw)
        return {"Opaque": rec}

    def test_root_out_of_range(self) -> None:
        j = self._rec(nodes=[{"Const": {"value": 1}}], root=5)
        with pytest.raises(HnlCodecError, match="root"):
            term_from_json(j)

    def test_root_not_int(self) -> None:
        j = self._rec(nodes=[{"Const": {"value": 1}}], root="0")
        with pytest.raises(HnlCodecError, match="root"):
            term_from_json(j)

    def test_nodes_not_list(self) -> None:
        j = self._rec(nodes={"0": {}}, root=0)
        with pytest.raises(HnlCodecError, match="nodes"):
            term_from_json(j)

    def test_arg_index_not_earlier(self) -> None:
        # node 0 references itself — impossible in post-order
        j = self._rec(
            nodes=[
                {
                    "Op": {
                        "name": "neg",
                        "args": [0],
                        "source": {"Foreign": "tensor"},
                    }
                }
            ],
            root=0,
        )
        with pytest.raises(HnlCodecError, match="earlier node index"):
            term_from_json(j)

    def test_arg_index_not_int(self) -> None:
        j = self._rec(
            nodes=[
                {"Const": {"value": 1}},
                {
                    "Op": {
                        "name": "neg",
                        "args": ["0"],
                        "source": {"Foreign": "tensor"},
                    }
                },
            ],
            root=1,
        )
        with pytest.raises(HnlCodecError, match="earlier node index"):
            term_from_json(j)

    def test_unknown_node_tag(self) -> None:
        j = self._rec(nodes=[{"Widget": {"x": 1}}], root=0)
        with pytest.raises(HnlCodecError, match="unknown tag"):
            term_from_json(j)

    def test_node_not_single_tagged_record(self) -> None:
        j = self._rec(
            nodes=[{"Const": {"value": 1}, "extra": 2}], root=0
        )
        with pytest.raises(HnlCodecError, match="tagged record"):
            term_from_json(j)


# ---------------------------------------------------------------------------
#  The ``attrs`` field — ADR-0173 forward spelling
# ---------------------------------------------------------------------------


class TestAttrsField:
    def test_attrs_field_decodes(self) -> None:
        j = {
            "Op": {
                "name": "clamp",
                "args": [],
                "attrs": {"min": {"__nonfinite__": "nan"}},
            }
        }
        back = term_from_json(j)
        assert isinstance(back, Op)
        _assert_nan(back.attrs["min"])

    def test_attrs_and_lit_ambiguous_declines(self) -> None:
        j = {
            "Op": {
                "name": "clamp",
                "args": [],
                "attrs": {"min": 0.0},
                "lit": {"__mlopt_attrs__": {"min": 1.0}},
            }
        }
        with pytest.raises(HnlCodecError, match="ambiguous"):
            term_from_json(j)

    def test_attrs_null_is_absent(self) -> None:
        j = {
            "Op": {
                "name": "neg",
                "args": [{"Const": {"value": 1}}],
                "attrs": None,
            }
        }
        assert term_from_json(j) == Op.make(
            "neg", Const(1), validate=False
        )

    def test_attrs_field_not_object_declines(self) -> None:
        j = {"Op": {"name": "neg", "args": [], "attrs": [1, 2]}}
        with pytest.raises(HnlCodecError, match="must be an object"):
            term_from_json(j)

    def test_lit_without_attrs_record_declines(self) -> None:
        j = {"Op": {"name": "neg", "args": [], "lit": {"other": 1}}}
        with pytest.raises(HnlCodecError, match="not an attrs record"):
            term_from_json(j)

    def test_attrs_record_not_object_declines(self) -> None:
        j = {
            "Op": {
                "name": "neg",
                "args": [],
                "lit": {"__mlopt_attrs__": 5},
            }
        }
        with pytest.raises(HnlCodecError, match="must be an object"):
            term_from_json(j)

    def test_bad_attrs_value_declines(self) -> None:
        # {"bogus": …} is neither __tuple__ nor __list__ — attr_from_data
        # refuses it and the codec surfaces the decline.
        j = {
            "Op": {
                "name": "neg",
                "args": [],
                "attrs": {"x": {"bogus": 1}},
            }
        }
        with pytest.raises(HnlCodecError, match="bad attrs payload"):
            term_from_json(j)
