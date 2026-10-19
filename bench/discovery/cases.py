"""Case loading — ``bench/discovery/cases/*.toml`` → typed records.

A case file looks like::

    name = "conv_bn_fold"
    title = "Conv2d + eval BatchNorm2d fold"
    category = "speed"              # memory|speed|structure|precision
    citation = "..."
    source = "conv_bn_eval"         # bench.discovery.fixtures.SOURCES key
    size = 6                        # the fixture's scale knob
    rules = "default"               # search() rules preset name
    cost = "launch"                 # symbolic axis — COST_FNS key
    seed = 0

    [expect]
    signals = ["law:conv_bn_fold"]  # any observed signal counts as found
    ops = ["conv2d"]                # required in the extracted term
    absent_ops = ["batch_norm"]     # forbidden in the extracted term

    [[baselines]]
    name = "eager"
    kind = "self"                   # the spelled module itself
    [[baselines]]
    name = "hand-folded conv"
    kind = "ref"                    # bench.discovery.refs.REFS key
    ref = "conv_bn_folded"
    [[baselines]]
    name = "inductor"
    kind = "compile"                # torch.compile — measured only
    [[baselines]]
    name = "FLA chunk_retention"
    kind = "external"               # outside the repo — honest empty row
    note = "scratch dep, ~7-11x ahead"

``signals`` are ``law:<rule-name>`` (the rule fired in saturation),
``stat:<stats-key>`` (a truthy non-local-pass record — the pairing /
lift passes are not rule fires), or ``op:<op-name>`` (present in the
extracted term).  ``found`` = at least one signal observed AND the
op spec holds on the extracted term.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

#: Case directory — ``bench/discovery/cases/``.
CASES_DIR = Path(__file__).resolve().parent / "cases"

CATEGORIES = ("memory", "speed", "structure", "precision")
BASELINE_KINDS = ("self", "ref", "compile", "external")
COST_AXES = ("launch", "flops", "count", "exec", "param_bytes")


@dataclass(frozen=True)
class Expect:
    """The derivability contract — what counts as the named transform."""

    signals: tuple[str, ...] = ()
    ops: tuple[str, ...] = ()
    absent_ops: tuple[str, ...] = ()
    #: ``{op: n}`` — the extracted term may hold at most ``n``
    #: occurrences of the op (the fused ``mul_unsq`` form lifts the
    #: view once; the spelled form lifts it twice).
    max_ops: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Baseline:
    """One registered comparison arm.

    ``kind``:

    * ``self`` — the spelled module, eager;
    * ``ref`` — a known-good implementation from
      :data:`bench.discovery.refs.REFS` (``ref`` names the key);
    * ``compile`` — ``torch.compile`` of the spelled module
      (measured-only: no symbolic form is registered);
    * ``external`` — a named baseline outside the repository; the row
      records it as unavailable so the floor is honestly bounded by
      what the suite can actually run.
    """

    name: str
    kind: str = "self"
    ref: str = ""
    note: str = ""


@dataclass(frozen=True)
class DiscoveryCase:
    """One named optimization the suite checks derivability for."""

    name: str
    title: str
    category: str
    citation: str
    source: str
    expect: Expect = field(default_factory=Expect)
    baselines: tuple[Baseline, ...] = ()
    size: int = 0
    rules: str = "default"
    cost: str = "launch"
    seed: int = 0
    max_iterations: int = 8
    max_enodes: int = 100_000
    notes: str = ""


def _parse_case(path: Path, data: dict) -> DiscoveryCase:
    """Validate one TOML payload into a :class:`DiscoveryCase`."""
    for key in ("name", "title", "category", "citation", "source"):
        if key not in data:
            raise ValueError(
                f"{path.name}: missing required key {key!r}"
            )
    if data["category"] not in CATEGORIES:
        raise ValueError(
            f"{path.name}: category {data['category']!r} not in "
            f"{CATEGORIES}"
        )
    if data.get("cost", "launch") not in COST_AXES:
        raise ValueError(
            f"{path.name}: cost {data['cost']!r} not in {COST_AXES}"
        )
    exp = data.get("expect", {})
    baselines = []
    for b in data.get("baselines", []):
        if b.get("kind", "self") not in BASELINE_KINDS:
            raise ValueError(
                f"{path.name}: baseline kind {b.get('kind')!r} not in "
                f"{BASELINE_KINDS}"
            )
        if b.get("kind") == "ref" and not b.get("ref"):
            raise ValueError(
                f"{path.name}: ref baseline {b.get('name')!r} "
                "needs a `ref` key"
            )
        baselines.append(
            Baseline(
                name=b["name"],
                kind=b.get("kind", "self"),
                ref=b.get("ref", ""),
                note=b.get("note", ""),
            )
        )
    return DiscoveryCase(
        name=data["name"],
        title=data["title"],
        category=data["category"],
        citation=data["citation"],
        source=data["source"],
        expect=Expect(
            signals=tuple(exp.get("signals", ())),
            ops=tuple(exp.get("ops", ())),
            absent_ops=tuple(exp.get("absent_ops", ())),
            max_ops={
                str(k): int(v)
                for k, v in exp.get("max_ops", {}).items()
            },
        ),
        baselines=tuple(baselines),
        size=int(data.get("size", 0)),
        rules=data.get("rules", "default"),
        cost=data.get("cost", "launch"),
        seed=int(data.get("seed", 0)),
        max_iterations=int(data.get("max_iterations", 8)),
        max_enodes=int(data.get("max_enodes", 100_000)),
        notes=data.get("notes", ""),
    )


def load_cases(cases_dir: Path | None = None) -> list[DiscoveryCase]:
    """Load every ``*.toml`` under *cases_dir*, sorted by name."""
    d = Path(cases_dir) if cases_dir is not None else CASES_DIR
    cases = []
    for path in sorted(d.glob("*.toml")):
        with path.open("rb") as fh:
            cases.append(_parse_case(path, tomllib.load(fh)))
    return cases
