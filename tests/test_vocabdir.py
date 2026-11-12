"""Plan 0033 phase 0 — the canonical fragment vocabulary.

``vocab/`` is the single source of truth for the discovery
vocabulary: six canonical JSON files (``ops``/``laws``/``handlers``/
``schemas``/``prims``/``families``, each ``schema:``-pinned) plus the
handl-facing projections (``ast.yaml``/``transforms.yaml`` and the
shell stubs) that ``mlopt-handl-peer``'s ``transform_ids(dir)`` reads.
These tests pin the round-trip: emit → load → the same objects the
Python registries hold, and the same dir ``pipeline.propose``
consumes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import Op, TensorType, Var
from mlopt_core.laws import ALL_RULES
from mlopt_core.laws.serialize import law_to_data
from mlopt_discovery import vocabdir

_REPO = Path(__file__).resolve().parents[1]
_VOCAB = _REPO / "vocab"

try:  # the shipping surface — absent until maturin builds it
    import mlopt_handl_peer as _ext
except ImportError:  # pragma: no cover - depends on the local build
    _ext = None


# ---------------------------------------------------------------------------
#  The directory itself
# ---------------------------------------------------------------------------


def test_vocab_dir_exists_with_schema_pins() -> None:
    """All six canonical files plus the handl projections are present."""
    assert _VOCAB.is_dir(), (
        "vocab/ missing — run scripts/emit_vocab_fragments.py"
    )
    for name in vocabdir.CANONICAL_FILES + vocabdir.PROJECTION_FILES:
        assert (_VOCAB / name).is_file(), name
    for name in vocabdir.CANONICAL_FILES:
        doc = json.loads((_VOCAB / name).read_text())
        assert doc["schema"] == vocabdir.SCHEMA_VERSION, name


def test_default_vocab_dir_resolves_the_repo_dir() -> None:
    """``default_vocab_dir`` finds ``vocab/`` from a repo checkout."""
    assert vocabdir.default_vocab_dir() == _VOCAB


def test_load_vocab_round_trips_the_banks() -> None:
    """The Python projection sees every law, schema, prim and op."""
    v = vocabdir.load_vocab(_VOCAB)
    assert len(v.laws) == len(ALL_RULES)
    assert {r.name for r in v.laws} == {r.name for r in ALL_RULES}
    assert len(v.schemas) == 5
    assert len(v.prims) == 58
    assert len(v.ops) == len(vocabdir.collect_ops())
    # conv_bn_fold's procedural hooks cannot serialize — the loader
    # re-binds them by name and *says so*.
    assert "conv_bn_fold" in v.hooked
    hooked = v.laws_by_name()["conv_bn_fold"]
    shipped = next(r for r in ALL_RULES if r.name == "conv_bn_fold")
    assert hooked.check is shipped.check


def test_handlers_cover_shipped_prim_and_minted() -> None:
    """handlers.yaml carries all three provenance classes."""
    v = vocabdir.load_vocab(_VOCAB)
    table = v.handler_table()
    for tag in ("silu", "square", "swiglu"):
        assert tag in table
        assert table[tag]["kernel"] == tag
    assert len(v.prim_handlers) == 58
    assert len(v.minted_handlers) == 53
    assert all(h["op"].startswith("synop") for h in v.prim_handlers)


def test_families_tables_match_the_grammar_constants() -> None:
    """families.yaml is the handler-grammar tables as data."""
    from mlopt_discovery import family_synth, handler_synth

    v = vocabdir.load_vocab(_VOCAB)
    fam = v.families
    assert fam["names"] == list(handler_synth._FAMILIES)
    assert fam["grammar_version"] == handler_synth._GRAMMAR_VERSION
    assert fam["broadcast_ops"] == sorted(handler_synth.BROADCAST_OPS)
    assert fam["orders"] == list(family_synth.ORDERS)
    assert set(fam["leaf_forms"]) == set(family_synth.LEAF_FORMS)


def test_check_vocab_is_fresh() -> None:
    """The committed directory passes its own drift gate."""
    assert vocabdir.check_vocab(_VOCAB) == []


# ---------------------------------------------------------------------------
#  Emit + upsert — workstream 0b's fragment-emit API
# ---------------------------------------------------------------------------


@pytest.fixture()
def vocab_copy(tmp_path: Path) -> Path:
    """A writable copy of the shipped vocabulary."""
    dst = tmp_path / "vocab"
    shutil.copytree(_VOCAB, dst)
    return dst


def test_emit_all_is_idempotent(vocab_copy: Path) -> None:
    """Re-emitting changes nothing — owner rows upsert in place."""
    before = {
        p.name: p.read_text()
        for p in vocab_copy.iterdir()
        if p.is_file()
    }
    vocabdir.emit_all(vocab_copy)
    after = {
        p.name: p.read_text()
        for p in vocab_copy.iterdir()
        if p.is_file()
    }
    assert before == after
    assert vocabdir.check_vocab(vocab_copy) == []


def _fake_prim_record(name: str) -> dict:
    """A minimal mint-shaped record for ``emit_prim``."""
    fold = Rewrite(
        f"prim:{name}",
        Op.make("add", "a", "b"),
        Op.make(name, "a", "b"),
    )
    return {
        "alpha": "(add 'a' 'b')",
        "sites": 2,
        "equal_sites": 2,
        "mods": ["m1"],
        "fold": law_to_data(fold),
        "handler": {
            "family": "ew",
            "max_err": 0.0,
            "record": {
                "tag": f"minted_{name}",
                "kind": "handler",
                "pattern": ["add", "a", "b"],
                "kernel": f"minted_{name}_k",
                "args": ["a", "b"],
                "handler_body": ["ew", ["add", "a", "b"]],
                "provenance": {"atol": 1e-12},
            },
        },
    }


def test_emit_prim_lands_op_handler_and_record(
    vocab_copy: Path,
) -> None:
    """0b: one ``emit_prim`` writes the op row, handler row, record."""
    name = "synop_0123456789"
    vocabdir.emit_prim(vocab_copy, name, _fake_prim_record(name))

    v = vocabdir.load_vocab(vocab_copy, fresh=True)
    assert name in v.op_names()
    assert v.op_row(name)["denotation"] == "prim"
    assert name in v.prims
    assert any(h["op"] == name for h in v.minted_handlers)
    # the projection saw the op too
    assert name in (vocab_copy / "ast.yaml").read_text()


def test_emit_preserves_machine_added_fragments(
    vocab_copy: Path,
) -> None:
    """A fragment added by the mint survives re-emission.

    The directory is the source and the registries its projection:
    the next ``emit_all`` (owner refresh) upserts owner rows but
    cannot wipe rows the owners do not know.
    """
    name = "synop_0123456789"
    vocabdir.emit_prim(vocab_copy, name, _fake_prim_record(name))

    vocabdir.emit_all(vocab_copy)
    v = vocabdir.load_vocab(vocab_copy, fresh=True)
    assert name in v.op_names()
    assert name in v.prims
    assert any(h["op"] == name for h in v.minted_handlers)


def test_emit_prim_writes_full_fragment_set(
    vocab_copy: Path,
) -> None:
    """Phase 3: ``emit_prim`` lands every fragment the mint owns.

    One call writes the op row, the ``prim:`` fold law, the minted
    handler row, the verbatim prim record, the ``denotation:`` /
    ``handler:`` model rows and the ``fold:`` schema + its
    ``.expand`` premise closure — then the directory still passes
    its own drift gate (machine rows are not owner rows).
    """
    name = "synop_0123456789"
    vocabdir.emit_prim(vocab_copy, name, _fake_prim_record(name))

    v = vocabdir.load_vocab(vocab_copy, fresh=True)
    assert name in v.op_names()
    assert name in v.prims
    assert f"prim:{name}" in v.laws_by_name()
    schema = f"fold:{name}"
    assert schema in {r.name for r in v.schemas}
    assert schema in v.fusion_premises
    model_ids = {m["id"] for m in v.models}
    assert f"denotation:{name}" in model_ids
    assert f"handler:{name}" in model_ids
    assert any(h["op"] == name for h in v.minted_handlers)
    assert vocabdir.check_vocab(vocab_copy) == []

    # an owner refresh cannot wipe the machine-added set.
    vocabdir.emit_all(vocab_copy)
    v2 = vocabdir.load_vocab(vocab_copy, fresh=True)
    assert name in v2.prims
    assert f"prim:{name}" in v2.laws_by_name()
    assert schema in {r.name for r in v2.schemas}
    assert schema in v2.fusion_premises
    assert {m["id"] for m in v2.models} >= model_ids


def test_emit_atom_writes_schema_fragments(
    vocab_copy: Path,
) -> None:
    """Phase 3 anti-unification arm: ``emit_atom`` lands schema rows.

    A mined ``AtomCandidate`` admitted through ``mint_atoms``
    becomes a ``fold:<atom>`` recognizer + ``.expand`` premise in
    ``schemas.yaml``, its ``intro`` law in ``laws.yaml`` and its op
    row in ``ops.yaml``.
    """
    from mlopt_core.egraph import EGraph
    from mlopt_discovery.antiunify import mint_atoms

    eg = EGraph()
    x, w, a = (
        Var(n, TensorType((4, 4))) for n in ("x", "w", "a")
    )
    terms = [
        Op.make("add", Op.make("mul", x, w), a),
        Op.make("add", Op.make("mul", a, w), x),
        Op.make("add", Op.make("mul", x, x), w),
    ]
    eg.add_term(Op.make("pack", *terms))
    (atom,) = mint_atoms(eg, count=1, prefix="vocabtest")

    vocabdir.emit_atom(vocab_copy, atom)

    v = vocabdir.load_vocab(vocab_copy, fresh=True)
    assert atom.name in v.op_names()
    assert f"{atom.name}.intro" in v.laws_by_name()
    schema = f"fold:{atom.name}"
    assert schema in {r.name for r in v.schemas}
    # the premise closure carries the unfold direction.
    premises = v.fusion_premises[schema]
    assert [p.name for p in premises] == [f"{atom.name}.expand"]
    assert vocabdir.check_vocab(vocab_copy) == []


def test_emit_family_rows_roundtrip(vocab_copy: Path) -> None:
    """Minted families + refused specs persist and decode back.

    ``emit_family_rows`` writes the ``families.minted_families`` /
    ``families.refused_specs`` machine sections in one sync;
    ``vocab_extra_families`` returns the descriptors in the live
    ``extra_families`` shape (tuples restored, ``covers`` dropped).
    """
    desc = {
        "name": "fam:probe",
        "gate": {"kind": "reduce"},
        "leaves": {"read": {"kind": "read", "attrs": ["axis"]}},
        "broadcast": ["neg"],
        "requires": [],
        "orders": ["read_then_map", "map_then_read"],
        "covers": ["synop_probe"],
        "provenance": {"budget": 4},
    }
    refusal = {
        "key": "synop_refused01",
        "op": "synop_refused01",
        "spec": {"spec": ("add", "a", "b"), "args": []},
        "domain": [{"a": [2, 4]}],
        "verdict": "refused",
        "tried": 3,
        "best_err": "inf",
        "note": "no body verified",
    }
    vocabdir.emit_family_rows(
        vocab_copy, families=[desc], refusals=[refusal]
    )

    v = vocabdir.load_vocab(vocab_copy, fresh=True)
    fams = v.families["minted_families"]
    assert [f["name"] for f in fams] == ["fam:probe"]
    assert fams[0]["covers"] == ["synop_probe"]
    refs = v.families["refused_specs"]
    assert [r["key"] for r in refs] == ["synop_refused01"]

    extra = vocabdir.vocab_extra_families(v)
    assert len(extra) == 1
    assert extra[0]["name"] == "fam:probe"
    assert "covers" not in extra[0]
    # tuples restored — the live-descriptor shape mint_handlers eats.
    assert isinstance(extra[0]["orders"], tuple)

    # owner refresh keeps the machine sections.
    vocabdir.emit_all(vocab_copy)
    v2 = vocabdir.load_vocab(vocab_copy, fresh=True)
    assert v2.families["minted_families"] == fams
    assert v2.families["refused_specs"] == refs
    assert vocabdir.check_vocab(vocab_copy) == []


def test_upsert_law_and_handler(vocab_copy: Path) -> None:
    """The row-level upserts land in their canonical files."""
    record = law_to_data(
        Rewrite(
            "zz_probe_law",
            Op.make("mul", "A", "B"),
            Op.make("mul", "B", "A"),
        )
    )
    vocabdir.upsert_law(vocab_copy, record)
    vocabdir.upsert_handler(
        vocab_copy,
        {
            "tag": "zz_probe",
            "source": "lawdata",
            "pattern": {
                "spec": "mul",
                "args": [{"mvar": "X"}, {"mvar": "X"}],
                "attrs": {},
            },
            "kernel": "square",
            "args": ["X"],
        },
    )
    v = vocabdir.load_vocab(vocab_copy, fresh=True)
    assert "zz_probe_law" in v.laws_by_name()
    assert v.handler_table()["zz_probe"]["kernel"] == "square"
    assert (
        "transform.mlopt.zz_probe_law"
        in (vocab_copy / "transforms.yaml").read_text()
    )


def test_hand_added_law_visible_to_both_engines(
    vocab_copy: Path,
) -> None:
    """The exit gate: a hand-added fragment reaches Python AND handl.

    Simulates a *hand* edit — a raw JSON row appended to ``laws.yaml``
    (no upsert API), then the projection resynced — and checks both
    consumers see ``transform.mlopt.zz_phase0_probe``: the Python
    loader's law bank and the YAML the peer's ``transform_ids`` reads.
    """
    record = law_to_data(
        Rewrite(
            "zz_phase0_probe",
            Op.make("add", "A", "B"),
            Op.make("add", "B", "A"),
            law="phase-0 probe — commutativity spelled backwards",
        )
    )
    laws_path = vocab_copy / "laws.yaml"
    doc = json.loads(laws_path.read_text())
    doc["laws"].append(record)
    laws_path.write_text(json.dumps(doc, indent=2) + "\n")
    vocabdir.sync_projections(vocab_copy)

    v = vocabdir.load_vocab(vocab_copy, fresh=True)
    assert "zz_phase0_probe" in v.laws_by_name()

    transforms = (vocab_copy / "transforms.yaml").read_text()
    assert "transform.mlopt.zz_phase0_probe" in transforms


def test_check_vocab_flags_projection_drift(vocab_copy: Path) -> None:
    """A stale ``transforms.yaml`` is drift, not silent staleness."""
    transforms = vocab_copy / "transforms.yaml"
    transforms.write_text(transforms.read_text() + "\n# stale\n")
    drifted = vocabdir.check_vocab(vocab_copy)
    assert "transforms.yaml" in drifted


def test_check_vocab_flags_owner_row_edits(vocab_copy: Path) -> None:
    """Editing an owner-backed row is drift (emit would revert it)."""
    ops_path = vocab_copy / "ops.yaml"
    doc = json.loads(ops_path.read_text())
    doc["ops"][0]["arity"] = 999
    ops_path.write_text(json.dumps(doc, indent=2) + "\n")
    drifted = vocabdir.check_vocab(vocab_copy)
    assert any(d.startswith("ops.yaml:ops:") for d in drifted)


# ---------------------------------------------------------------------------
#  pipeline.propose consumes the same dir
# ---------------------------------------------------------------------------


def _tiny_corpus() -> tuple[dict, list]:
    """A minimal census + terms exercising the fusion-schema arm."""
    from mlopt_discovery.census import CorpusTerm, op_tuple_census

    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    terms = [
        Op.make("mul", x, Op.make("sigmoid", x)),
        Op.make("add", Op.make("mul", x, y), y),
    ]
    cts = [CorpusTerm("test", "t", t) for t in terms]
    counts, _ = op_tuple_census(cts)
    return dict(counts), terms


def _sig(props: list) -> list[tuple]:
    """A comparison signature for proposal pools (instance-free)."""
    from mlopt_discovery import proposal as lp

    return sorted(
        (p.name, p.family, p.sources, lp._key(p.lhs, p.rhs))
        for p in props
    )


def test_propose_vocab_dir_parity() -> None:
    """``propose`` over the dir proposes what the Python tables do.

    The fusion-schema arm is the only consumer the directory feeds
    today; every other proposer is untouched, so the pools must be
    identical — the dir is a *projection* of the same facts.
    """
    from mlopt_discovery import pipeline as pipe

    census, terms = _tiny_corpus()
    off = pipe.propose(census, terms, "hand", vocab_dir=False)
    on = pipe.propose(census, terms, "hand")  # auto → vocab/
    assert vocabdir.default_vocab_dir() == _VOCAB
    assert _sig(off) == _sig(on)


def test_propose_explicit_vocab_dir_and_bad_dir(tmp_path: Path) -> None:
    """``vocab_dir=<path>`` loads that dir; a non-dir refuses loudly."""
    from mlopt_discovery import pipeline as pipe

    census, terms = _tiny_corpus()
    props = pipe.propose(census, terms, "hand", vocab_dir=_VOCAB)
    assert isinstance(props, list)
    with pytest.raises(FileNotFoundError):
        pipe.propose(census, terms, "hand", vocab_dir=tmp_path)


# ---------------------------------------------------------------------------
#  The exit gate — handl loads the same dir propose consumes
# ---------------------------------------------------------------------------


@pytest.mark.e2e
def test_handl_peer_loads_the_vocab_dir() -> None:
    """``transform_ids(vocab/)`` sees the whole law bank.

    The peer's ``language_transform_ids`` is the handl-side read of
    the same directory ``pipeline.propose`` consumes — plan 0033
    phase 0's exit gate.  Needs the maturin extension (the CLI
    binary exposes only ``rewrite``, not the ids listing); skips
    without it, like ``test_handl_peer.py``.
    """
    if _ext is None:
        pytest.skip(
            "mlopt_handl_peer extension not built "
            "(see test_handl_peer.py docstring)"
        )
    ids = _ext.language_transform_ids(str(_VOCAB))
    v = vocabdir.load_vocab(_VOCAB)
    expected = {f"transform.mlopt.{r.name}" for r in v.laws}
    assert expected <= set(ids), (
        f"{len(expected - set(ids))} law ids missing from handl load"
    )
