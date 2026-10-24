"""Tests for ``mlopt_discovery.family_synth`` — the family meta-grammar.

Pins the descriptor machinery (the generic ``("family", desc, sub)``
eval arm is the ``gather`` family's own translation, parameterised —
the parity is asserted, not assumed), the ``mint_families``
verify-or-refuse contract (a minted family must reproduce ≥1 spec
fp64 — mint-a-label is refused), and the injection seam (a minted
descriptor serves the specs the shipped families refuse through
``extra_families``).

The flagship row: the four ``select``/``linear``-bearing specs the
grammar-v3 prim mint refused (the ``prim_ops.*.handler`` refusal rows
of ``docs/reports/results_overnight_prim_v3.json``, embedded here
verbatim — spec + bound-site domain) recover a ``gather``-equivalent
descriptor covering all four, plus a narrower ``{select}``-gate
sub-family for the select×select spec.
"""

import json

import pytest
import torch

import mlopt_torch.torch_bridge as tb
from mlopt_discovery import family_synth as fs
from mlopt_discovery import handler_synth as hs

BINDINGS = tb._IR_TO_TORCH

#: The descriptor the ``gather`` family hand-codes — its leaf
#: vocabulary, broadcast views, marker and orders as data.  The
#: parity tests assert the generic arm reproduces ``_gather_view`` /
#: ``_gather_marked`` / ``_gather_bindings`` exactly.
GATHER_DESC = {
    "name": "gather-equiv",
    "gate": ["linear", "select"],
    "leaves": {
        "at": {
            "kind": "read",
            "from": "select",
            "binds": "select",
            "attrs": ("dim", "index"),
        },
        "aff": {"kind": "map", "from": "linear", "binds": "linear"},
    },
    "broadcast": ["squeeze", "unsqueeze"],
    "requires": "at",
    "orders": list(fs.ORDERS),
}

#: The four grammar-v3 refusal specs — embedded verbatim from
#: ``docs/reports/results_overnight_prim_v3.json`` prim_ops rows
#: (spec + bound-site domain), metavar names preserved.
V3_REFUSALS = [
    {  # prim:synop_5f1bba5798 — select-gate × select-of-linear
        "spec": (
            "mul",
            ("unsqueeze", ("select", "v430", {"dim": 1, "index": 0}),
             {"dim": -1}),
            ("unsqueeze",
             ("select", ("linear", "v463", "v464", "v465"),
              {"dim": 1, "index": 0}),
             {"dim": -1}),
        ),
        "domain": [
            {"v430": [2, 4, 1], "v463": [2, 4, 16],
             "v464": [16, 16], "v465": [16]}
        ]
        * 4,
    },
    {  # prim:synop_4c657cb2c0 — residual + broadcast select-of-linear
        "spec": (
            "add",
            "v13",
            ("mul",
             ("unsqueeze", "v419", {"dim": -1}),
             ("unsqueeze",
              ("select", ("linear", "v633", "v634", "v635"),
               {"dim": 1, "index": 7}),
              {"dim": 1})),
        ),
        "domain": [
            {"v13": [2, 16, 4], "v419": [2, 16], "v633": [2, 8, 16],
             "v634": [4, 16], "v635": [4]}
        ]
        * 4,
    },
    {  # prim:synop_00cd48d9a2 — gate × select-of-linear
        "spec": (
            "mul",
            ("unsqueeze", "v319", {"dim": -1}),
            ("unsqueeze",
             ("select", ("linear", "v605", "v606", "v607"),
              {"dim": 1, "index": 0}),
             {"dim": -1}),
        ),
        "domain": [
            {"v319": [2, 1], "v605": [2, 4, 16],
             "v606": [16, 16], "v607": [16]}
        ]
        * 4,
    },
    {  # prim:synop_336e34145b — select × select
        "spec": (
            "mul",
            ("unsqueeze", ("select", "v430", {"dim": 1, "index": 0}),
             {"dim": -1}),
            ("unsqueeze", ("select", "v458", {"dim": 1, "index": 0}),
             {"dim": -1}),
        ),
        "domain": [{"v430": [2, 4, 1], "v458": [2, 4, 16]}] * 4,
    },
]


def _env(shapes: dict, seed: int = 0) -> dict:
    return hs._sample_env(shapes, seed)


# ---------------------------------------------------------------------------
#  The generic ("family", desc, sub) eval arm — gather, parameterised
# ---------------------------------------------------------------------------


class TestFamilyArm:
    def test_eval_body_family_descriptor(self):
        """The generic arm evaluates a descriptor's leaf tree through
        the ambient bindings — the leaves resolve via ``binds``."""
        env = _env({"X": (2, 4, 8), "W": (4, 8), "B": (4,)})
        body = (
            "family",
            GATHER_DESC,
            ("aff", ("at", "X", {"dim": 1, "index": 0}), "W", "B"),
        )
        out = hs.eval_body(body, env, BINDINGS)
        torch.testing.assert_close(
            out, torch.nn.functional.linear(env["X"][:, 0], env["W"], env["B"])
        )

    def test_family_bindings_generalise_gather_bindings(self):
        """The descriptor's leaf table equals the hand-written
        ``_gather_bindings`` — ``at`` re-seats ``select``, ``aff``
        ``linear``."""
        gen = hs._family_bindings(BINDINGS, GATHER_DESC)
        hand = hs._gather_bindings(BINDINGS)
        env = _env({"X": (2, 4)})
        torch.testing.assert_close(
            gen["at"](env["X"], dim=1, index=0),
            hand["at"](env["X"], dim=1, index=0),
        )
        assert gen["aff"] is hand["aff"]

    def test_family_leaf_bind_missing_raises_loudly(self):
        """A leaf naming an ambient op the table lacks fails at eval
        — a malformed descriptor, never a silent degrade."""
        desc = {
            **GATHER_DESC,
            "leaves": {"zz": {"kind": "read", "from": "select",
                              "binds": "no_such_op",
                              "attrs": ("dim", "index")}},
        }
        env = _env({"X": (2, 4)})
        with pytest.raises(KeyError):
            hs.eval_body(
                ("family", desc, ("zz", "X", {"dim": 1, "index": 0})),
                env,
                BINDINGS,
            )


class TestDescriptorTranslation:
    """``_family_translate``/``_family_marked``/``_family_seeds`` are
    ``_gather_view``/``_gather_marked``/``_gather_seeds`` with the
    descriptor in place of the hard-coded vocabulary — the parity is
    asserted per spec."""

    @pytest.mark.parametrize("item", V3_REFUSALS)
    def test_seeds_parity_with_gather(self, item):
        gen = [b[2] for b in hs._family_seeds(item["spec"], GATHER_DESC)]
        hand = [b[1] for b in hs._gather_seeds(item["spec"])]
        assert gen == hand

    def test_both_order_seeds_distinct_when_fusable(self):
        """A select-of-affine yields the fused then the denotation
        seed — ``aff(at(x))`` before ``at(aff(x))``."""
        spec = ("select", ("linear", "X", "W", "B"),
                {"dim": 1, "index": 0})
        bodies = hs._family_seeds(spec, GATHER_DESC)
        assert bodies[0][2] == (
            "aff", ("at", "X", {"dim": 1, "index": 0}), "W", "B"
        )
        assert bodies[1][2] == (
            "at", ("aff", "X", "W", "B"), {"dim": 1, "index": 0}
        )

    def test_marker_gate_no_index_read(self):
        """A tree with no ``at`` leaf is not claimed — requires="at"
        keeps the no-tautology gate (``_gather_marked``'s own bar)."""
        assert hs._family_seeds(
            ("mul", "A", ("unsqueeze", "B", {"dim": -1})), GATHER_DESC
        ) == []
        # pure-ew spec — the ``ew`` replay's claim, never this family's
        assert hs._family_seeds(("mul", "A", "B"), GATHER_DESC) == []

    def test_marker_requires_specific_leaf(self):
        """requires="aff" claims only affine-bearing trees — the
        select×select spec stays outside that family."""
        desc = {**GATHER_DESC, "requires": "aff"}
        spec = V3_REFUSALS[3]["spec"]  # select×select — no linear
        assert hs._family_seeds(spec, desc) == []
        assert hs._family_seeds(V3_REFUSALS[0]["spec"], desc)

    def test_marker_star_any_leaf(self):
        """requires="*" admits any leaf-bearing tree."""
        desc = {**GATHER_DESC, "requires": "*"}
        assert hs._family_seeds(V3_REFUSALS[3]["spec"], desc)

    def test_gate_bounds_the_claim(self):
        """An op outside the gate stays untranslatable — a
        select-only family refuses a select+linear spec."""
        desc = {
            **GATHER_DESC,
            "gate": ["select"],
            "leaves": {"at": GATHER_DESC["leaves"]["at"]},
        }
        assert hs._family_seeds(V3_REFUSALS[0]["spec"], desc) == []
        # …but claims the select×select spec — a narrower covering family
        assert hs._family_seeds(V3_REFUSALS[3]["spec"], desc)

    def test_broadcast_subset_bounds_views(self):
        """A descriptor without ``unsqueeze`` in its broadcast set
        cannot translate the unsqueeze-bearing specs."""
        desc = {**GATHER_DESC, "broadcast": ["squeeze"]}
        assert hs._family_seeds(V3_REFUSALS[0]["spec"], desc) == []
        desc = {**GATHER_DESC, "broadcast": []}
        assert hs._family_seeds(V3_REFUSALS[0]["spec"], desc) == []

    def test_read_requires_literal_attrs(self):
        """A ``select`` without literal ``dim``/``index`` cannot
        translate — the read carries the spec's own coordinates."""
        spec = ("mul", ("select", "X", {"dim": 1}), "Y")
        assert hs._family_seeds(spec, GATHER_DESC) == []
        # no attr dict at all, or extra args — same refusal
        assert hs._family_seeds(("mul", ("select", "X"), "Y"),
                                GATHER_DESC) == []
        assert hs._family_seeds(
            ("mul", ("select", "X", "Y", {"dim": 1, "index": 0}), "Z"),
            GATHER_DESC,
        ) == []

    def test_read_source_may_be_broadcast_view(self):
        """A broadcast view of a leaf is a legal read source — the
        ``_gather_leaf`` rule parameterised (view-of-leaf, not
        composite)."""
        spec = (
            "mul",
            ("select", ("unsqueeze", "X", {"dim": 0}),
             {"dim": 1, "index": 0}),
            "Y",
        )
        seeds = hs._family_seeds(spec, GATHER_DESC)
        assert seeds[0][2] == (
            "mul",
            ("at", ("unsqueeze", "X", {"dim": 0}),
             {"dim": 1, "index": 0}),
            "Y",
        )
        # a broadcast view of a COMPOSITE is not a leaf — refusal
        spec2 = (
            "mul",
            ("select", ("unsqueeze", ("mul", "A", "B"), {"dim": 0}),
             {"dim": 1, "index": 0}),
            "Y",
        )
        assert hs._family_seeds(spec2, GATHER_DESC) == []

    def test_const_leaf_passthrough(self):
        """Spec constants ride the translated tree untouched — a
        gather tree may carry spec consts (``_gather_view``'s own
        passthrough)."""
        spec = (
            "mul",
            ("select", "X", {"dim": 1, "index": 0}),
            2.0,
        )
        seeds = hs._family_seeds(spec, GATHER_DESC)
        assert seeds[0][2] == (
            "mul", ("at", "X", {"dim": 1, "index": 0}), 2.0
        )

    def test_gate_op_without_leaf_rule_unclaimed(self):
        """A gate naming an op no leaf rule re-seats translates
        nothing — the gate is honest about what it cannot spell."""
        desc = {
            **GATHER_DESC,
            "leaves": {"at": GATHER_DESC["leaves"]["at"]},
        }
        # gate still claims linear, but no leaf re-seats it
        assert hs._family_seeds(V3_REFUSALS[0]["spec"], desc) == []

    def test_unknown_leaf_kind_falls_through(self):
        """A leaf rule whose ``kind`` is neither ``read`` nor ``map``
        matches no translation arm — the spec falls out of the
        family rather than being silently re-spelled."""
        desc = {
            **GATHER_DESC,
            "leaves": {
                "at": {"kind": "view", "from": "select",
                       "binds": "select"},
            },
        }
        assert hs._family_seeds(V3_REFUSALS[3]["spec"], desc) == []

    def test_read_of_composite_refused(self):
        """A select over a composite elementwise tree is the family's
        declared gap — the read's source must be a view leaf."""
        spec = (
            "mul",
            ("select", ("mul", "A", "B"), {"dim": 1, "index": 0}),
            "C",
        )
        assert hs._family_seeds(spec, GATHER_DESC) == []
        hand = hs._gather_seeds(spec)
        assert hand == []

    def test_map_over_composite_refused(self):
        """``linear`` of composite args is the contract family's
        claim — the map leaf takes metavar args only."""
        spec = (
            "mul",
            ("select", ("linear", ("mul", "A", "B"), "W", "B2"),
             {"dim": 1, "index": 0}),
            "C",
        )
        assert hs._family_seeds(spec, GATHER_DESC) == []

    def test_descriptor_bodies_are_plain_data(self):
        """Every emitted ``("family", desc, sub)`` candidate is
        JSON-able — the descriptor is provenance data."""
        for item in V3_REFUSALS:
            for body in hs._family_seeds(item["spec"], GATHER_DESC):
                json.dumps(hs.body_to_data(body))


# ---------------------------------------------------------------------------
#  mint_families — the meta-search, verify-or-refuse
# ---------------------------------------------------------------------------


class TestMintFamilies:
    def test_recovers_gather_equivalent(self):
        """The four v3 refusals mint a gather-equivalent descriptor:
        gate {linear, select}, leaves at:select + aff:linear,
        broadcast views admitted, marker ``at`` — covering all four
        at the fp64 gate."""
        fams, report = fs.mint_families(V3_REFUSALS)
        assert report["movable"] == ["linear", "select"]
        assert report["n_specs"] == 4
        assert len(report["coverage"]) == 4
        covering = [
            f
            for f in fams
            if len(f["provenance"]["covered"]) == 4
            and f["requires"] == "at"
        ]
        assert covering, "no marker-at family covered all four specs"
        fam = next(
            f for f in covering
            if sorted(f["broadcast"]) == ["squeeze", "unsqueeze"]
        )
        assert fam["gate"] == ["linear", "select"]
        assert fam["leaves"]["at"]["binds"] == "select"
        assert fam["leaves"]["aff"]["binds"] == "linear"
        prov = fam["provenance"]
        assert prov["origin"] == "minted-family"
        assert prov["grammar"] == hs._GRAMMAR_VERSION
        assert prov["max_err"] < 1e-9
        # every verified spec went through the fused order — the
        # strong claim, not the rename-equivalent fallback.
        assert prov["orders_verified"] == ["read_then_map"]

    def test_mints_distinct_sub_family(self):
        """The select-only spec is *also* covered by a {select}-gate
        family — a distinct covering family the enumeration finds
        first (smaller gate); the report records both."""
        fams, report = fs.mint_families(V3_REFUSALS)
        sel_only = [f for f in fams if f["gate"] == ["select"]]
        assert sel_only
        assert all(f["provenance"]["covered"] == [3] for f in sel_only)
        # first-coverer attribution: the select-only family wins spec 3
        assert report["coverage"]["3"] == sel_only[0]["name"]

    def test_minted_family_serves_the_refused_specs(self):
        """The seam: minted descriptors injected as extra_families
        mint all four specs at tried=1 — the same bodies v4's
        hand-written gather minted."""
        fams, _ = fs.mint_families(V3_REFUSALS)
        handlers, report = hs.mint_handlers(
            V3_REFUSALS, families=(), extra_families=tuple(fams)
        )
        assert len(handlers) == 4
        for row in report:
            assert row["verdict"] == "minted"
            assert row["family"].startswith("family:fam_")
            assert row["tried"] == 1
            assert row["max_err"] < 1e-9
            body = handlers[row["tag"]]["handler_body"]
            assert body[0] == "family"
            prov = handlers[row["tag"]]["provenance"]
            assert prov["family"].startswith("family:fam_")

    def test_minted_body_evaluates_and_delivers(self):
        """The injected family's minted atom runs through
        ``minted_bindings`` — the delivered callable, not a replay."""
        fams, _ = fs.mint_families(V3_REFUSALS)
        handlers, _ = hs.mint_handlers(
            [V3_REFUSALS[0]],
            families=(),
            extra_families=tuple(fams),
        )
        tag = next(iter(handlers))
        bindings = hs.minted_bindings(handlers)
        env = _env(V3_REFUSALS[0]["domain"][0])
        got = bindings[handlers[tag]["kernel"]](
            env["v430"], env["v463"], env["v464"], env["v465"]
        )
        ref = hs._eval_spec(V3_REFUSALS[0]["spec"], env, BINDINGS)
        torch.testing.assert_close(got, ref)

    def test_no_claim_for_untranslatable_ops(self):
        """A spec whose movable ops no leaf form re-seats is
        outside the whole descriptor space — every candidate is
        no-claim, a measured refusal."""
        items = [
            {
                "spec": ("matmul", "A", "B"),
                "domain": [{"A": (3, 4), "B": (4, 5)}],
            }
        ]
        fams, report = fs.mint_families(items)
        assert fams == []
        assert all(r["verdict"] == "no-claim" for r in report["families"])
        assert report["coverage"] == {}

    def test_pure_ew_spec_never_claimed(self):
        """An elementwise spec translates to a leaf-less tree — the
        marker gate refuses it everywhere (no tautology mints)."""
        items = [
            {
                "spec": ("mul", "X", ("sigmoid", "X")),
                "domain": [{"X": (4,)}],
            }
        ]
        fams, report = fs.mint_families(items)
        assert fams == []
        assert report["movable"] == []
        assert report["families"] == []

    def test_verify_refusal_measured(self, monkeypatch):
        """A family that translates a spec but cannot reproduce it is
        refused with the measured detail — never minted.

        With the shipped leaf inventory the ``map_then_read`` order
        is rename-equivalent (a re-seat verifies by construction), so
        the exercised refusal is a leaf bound to the *wrong* ambient
        op: ``at`` re-seating ``sum`` accepts the index attrs but
        reduces instead of reading — a same-shape wrong value, the
        measured ``numeric miss``, recorded."""
        bad = dict(fs.LEAF_FORMS["at"])
        bad["binds"] = "sum"
        monkeypatch.setitem(fs.LEAF_FORMS, "at", bad)
        fams, report = fs.mint_families(
            [V3_REFUSALS[3]], bindings=BINDINGS
        )
        assert fams == []
        refused = [
            r for r in report["families"] if r["verdict"] == "refused"
        ]
        assert refused
        assert all(
            0.0 < s["best_err"] < float("inf")
            for r in refused
            for s in r["specs"]
            if s["verdict"] == "refused"
        )

    def test_verify_refusal_raised_body(self, monkeypatch):
        """The louder refusal: a leaf bound to an ambient op that
        raises on the index attrs — ``at`` re-seating pointwise
        ``neg`` — is refused with ``inf``, never a quiet pass."""
        bad = dict(fs.LEAF_FORMS["at"])
        bad["binds"] = "neg"
        monkeypatch.setitem(fs.LEAF_FORMS, "at", bad)
        fams, report = fs.mint_families(
            [V3_REFUSALS[0]], bindings=BINDINGS
        )
        assert fams == []
        assert any(
            s["best_err"] == float("inf")
            for r in report["families"]
            for s in r["specs"]
            if s["verdict"] == "refused"
        )

    def test_fused_flag_marks_measured_content(self):
        """The verified row records whether the mint rests on the
        fused map∘read reorder (measured) or a rename-equivalent
        composition (by construction).  The select×select spec has no
        affine to fuse — its mint is honestly ``fused=False``."""
        fams, _ = fs.mint_families(V3_REFUSALS)
        fam = next(
            f for f in fams if len(f["provenance"]["covered"]) == 4
        )
        assert fam["provenance"]["fused"] == [0, 1, 2]

    def test_budget_cap_recorded(self):
        """A budget smaller than the search records the cap — a
        measured refusal, not a truncated guess."""
        _, report = fs.mint_families(V3_REFUSALS, budget=2)
        assert report["searched"] == 2
        assert report["budget_hit"] is True

    def test_unevaluable_spec_recorded(self):
        """A spec whose spelled eval raises is recorded unevaluable —
        no family can claim what cannot be drawn."""
        item = {
            "spec": ("select", "X", {"dim": 0, "index": 0}),
            "domain": [{"X": ()}],  # select on a 0-dim raises
        }
        fams, report = fs.mint_families([item])
        assert fams == []
        assert all(
            s["verdict"] == "unevaluable"
            for r in report["families"]
            for s in r["specs"]
        )


# ---------------------------------------------------------------------------
#  The descriptor is data — naming, serialisation, injection seam
# ---------------------------------------------------------------------------


class TestDescriptorData:
    def test_descriptor_name_is_deterministic(self):
        a = fs.descriptor_name(GATHER_DESC)
        b = fs.descriptor_name(dict(GATHER_DESC))
        assert a == b and a.startswith("fam_")
        assert a != fs.descriptor_name(
            {**GATHER_DESC, "requires": "*"}
        )

    def test_minted_family_is_jsonable(self):
        fams, _ = fs.mint_families(V3_REFUSALS)
        for fam in fams:
            json.dumps(fs.family_record(fam))

    def test_family_record_roundtrips_through_body_codec(self):
        """A minted descriptor survives the body's own data codec —
        ``body_from_data`` folds its lists back to tuples and the
        descriptor still serves seeds."""
        fams, _ = fs.mint_families(V3_REFUSALS)
        fam = next(
            f for f in fams if len(f["provenance"]["covered"]) == 4
        )
        rec = fs.family_record(fam)
        back = hs.body_from_data(json.loads(json.dumps(rec)))
        seeds = hs._family_seeds(V3_REFUSALS[3]["spec"], back)
        assert seeds and seeds[0][0] == "family"

    def test_extra_families_seed_order(self):
        """Injected descriptors propose their seeds ahead of the
        column walk — a minted family wins at tried=1, before any
        built-in column candidate."""
        bodies = [
            b
            for _, b in zip(
                range(4),
                hs.propose_bodies(
                    V3_REFUSALS[0]["spec"],
                    families=(),
                    extra_families=(GATHER_DESC,),
                ),
            )
        ]
        assert bodies[0][0] == "family"
        assert bodies[0][1] is GATHER_DESC

    def test_meta_less_spec_skips_injected_seeds(self):
        """The injected seam honours the metas gate — a spec with no
        metavar leaves proposes no descriptor seeds."""
        spec = ("select", 1.0, {"dim": 0, "index": 0})
        assert not any(
            b[0] == "family"
            for b in hs.propose_bodies(
                spec, families=(), extra_families=(GATHER_DESC,)
            )
        )

    def test_body_family_names_the_descriptor(self):
        body = ("family", {"name": "fam_x"}, "X")
        assert hs._body_family(body) == "family:fam_x"
        assert hs._body_family(("gather", "X")) == "gather"
        anon = ("family", {"leaves": {}}, "X")
        assert hs._body_family(anon) == "family:anon"


class TestSearchSpace:
    def test_movable_ops_inventory(self):
        assert fs._movable_ops(V3_REFUSALS[0]["spec"]) == {
            "select",
            "linear",
        }
        assert fs._movable_ops(("matmul", "A", "B")) == {"matmul"}
        assert fs._movable_ops(("mul", "A", "B")) == set()

    def test_gate_without_leaf_form_skipped(self):
        """A gate containing an op no leaf form re-seats produces no
        candidates — the gate could never translate it."""
        cands = list(fs._candidates(["matmul"]))
        assert cands == []
        cands = list(fs._candidates(["matmul", "select"]))
        assert all("matmul" not in c["gate"] for c in cands)

    def test_candidates_smallest_gate_first(self):
        cands = list(fs._candidates(["linear", "select"]))
        assert cands[0]["gate"] in (["linear"], ["select"])
        assert cands[0]["requires"] in set(cands[0]["leaves"]) | {"*"}
        assert len(cands) == 28

    def test_fused_composition_detects_map_of_read(self):
        desc = GATHER_DESC
        assert fs._fused_composition(
            ("aff", ("at", "X", {"dim": 1, "index": 0}), "W", "B"),
            desc,
        )
        assert not fs._fused_composition(
            ("at", ("aff", "X", "W", "B"), {"dim": 1, "index": 0}),
            desc,
        )
        assert not fs._fused_composition(
            ("mul", ("at", "X", {"dim": 1, "index": 0}), "Y"), desc
        )
        assert not fs._fused_composition("X", desc)
