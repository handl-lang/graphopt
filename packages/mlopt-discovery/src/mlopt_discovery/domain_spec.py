"""Domain-spec files — Stage-2 self-driving, first cut: domains-as-data.

Stage 1 made the *corpus* mechanical (:mod:`corpus_cov` generates
cases from measured coverage gaps; ``overnight --auto-admit`` plus
eqsat's ``AUTO_TIER`` admit verified laws without a human).  Stage 2
makes the *domain* mechanical: a spec file names a vocabulary
subset, a corpus selection, a ruleset profile and an incumbent
recipe, and :func:`register_file` emits a whole playable
:class:`~mlopt_discovery.play.Domain` — cases generator, board
factory, player spec and scripted arm — into ``play.DOMAINS``.

First cut scope (honest):

* ``board`` — only ``"meta-arena"`` is supported: the symbolic
  probe-tuple corpus (``(name, term, rules, decls, supported)``)
  played on :class:`~mlopt_discovery.meta_arena.MetaArena` under
  ``count_cost``.  The torch-backed boards (``torch``/``gen``)
  need module builders and sink bindings a JSON spec cannot yet
  name — the schema reserves ``board`` for them.
* ``vocab.ops`` / ``vocab.attrs`` — whitelists applied twice:
  per-kind *footprints* (sampled op/attr sets per builder kind)
  decide which kinds may draw at all, and a per-case post-filter
  enforces the invariant on every emitted term (a kind whose rare
  draws escape the footprint is dropped, not padded).
* ``vocab.handler_tags`` — a subset of ``lawdata.HANDLERS`` seated
  on the board (``MetaArena(handlers=…)``); unlisted tags are
  never claimed.
* ``vocab.rules`` — ``"all"`` (default), ``"default"`` (the case
  ruleset's names ⊆ ``DEFAULT`` — excludes the scan-extended
  cases), ``"scan"`` (⊆ the 154-law scan ruleset), or an explicit
  list of rule names.
* ``corpus.kinds`` / ``corpus.coverage`` — forwarded to
  :func:`meta_player.gen_cases`'s ``kinds``/``coverage`` seam; the
  effective draw is the intersection with the vocab-eligible set.
* ``shapes`` / ``hardware`` / ``dtype`` / ``device`` — **recorded,
  not consumed** in v1: the meta-arena terms are symbolic (fixed
  ``(4, 4)`` vars in ``gen_cases``), so there is no tensor shape or
  device to bind yet.  The fields exist so a torch-backed board
  can honour them later without a schema break.
* ``player`` — the learned arm's :class:`gamekit.players.DomainSpec`
  selector; only ``"meta"`` (``meta_player.META_SPEC``) exists.
* ``incumbent`` — the scripted/baseline recipe: ``"scripted"``
  (``MetaArena.ScriptedPlayer`` — saturate→extract), ``"greedy"``
  (``GreedyPlayer`` — cheapest-immediate), ``"none"``.

Spec files are JSON (stdlib — no new dependency).  Packaged specs
live in ``mlopt_discovery/domains/*.json`` and resolve by bare
name; ``--domain-spec`` on the ``play`` CLI accepts a path or a
packaged name::

    .venv/bin/python -m mlopt_discovery.play \
        --domain-spec pointwise --domain pointwise \
        --train-cases 8 --eval-cases 4

Run the module directly for a spec census (no episodes)::

    .venv/bin/python -m mlopt_discovery.domain_spec --spec pointwise
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlopt_core.cost.basic import count_cost
from mlopt_core.ir import Op
from mlopt_core.laws import DEFAULT

from . import corpus_cov, lawdata
from . import meta_arena as ma
from . import meta_player as mp

__all__ = [
    "BOARDS",
    "INCUMBENTS",
    "PLAYERS",
    "SPECS",
    "DomainSpecFile",
    "load_spec",
    "register_file",
    "spec_summary",
]

_DOMAINS_DIR = Path(__file__).resolve().parent / "domains"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_SAMPLE_CASES = 240  # footprint census size per spec build
_MAX_ROUNDS = 8  # post-filter oversampling rounds in ``cases``


# ---------------------------------------------------------------------------
#  The spec record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainSpecFile:
    """A parsed domain-spec file — validated data, not code.

    See the module docstring for the schema.  ``raw`` keeps the
    original mapping for fields v1 records but does not consume
    (``shapes``/``hardware``).
    """

    name: str
    board: str = "meta-arena"
    corpus_kinds: tuple[str, ...] | None = None
    corpus_coverage: bool = True
    vocab_ops: frozenset[str] | None = None
    vocab_attrs: frozenset[str] | None = None  # "op.attr" spellings
    handler_tags: tuple[str, ...] | None = None
    rules_profile: Any = "all"  # "all"|"default"|"scan"|tuple[name,...]
    player: str = "meta"
    incumbent: str = "scripted"
    shapes: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    description: str = ""
    raw: dict = field(default_factory=dict, compare=False)


# ---------------------------------------------------------------------------
#  Loading + validation
# ---------------------------------------------------------------------------


def _as_str_list(value: Any, *, where: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(v, str) for v in value
    ):
        raise ValueError(
            f"{where}: expected a list of strings, got {value!r}"
        )
    return tuple(value)


def _parse_rules(value: Any) -> Any:
    """Resolve ``vocab.rules`` to a profile name or a name tuple."""
    if value is None or value == "all":
        return "all"
    if value in ("default", "scan"):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, str) for v in value
    ):
        return tuple(value)
    raise ValueError(
        f"vocab.rules: expected 'all'|'default'|'scan'|[names], "
        f"got {value!r}"
    )


def load_spec(source: str | Path | dict) -> DomainSpecFile:
    """Parse a spec from a JSON path, packaged name, or mapping.

    A bare name (``pointwise``) resolves to the packaged
    ``domains/<name>.json``; anything path-like is read as JSON.
    """
    if isinstance(source, dict):
        data = dict(source)
    else:
        text = str(source)
        path = Path(text)
        if not path.exists() and not text.endswith(".json"):
            cand = _DOMAINS_DIR / f"{text}.json"
            if cand.exists():
                path = cand
        if not path.exists():
            raise ValueError(f"domain spec not found: {source!r}")
        data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("domain spec: top level must be an object")
    return _spec_from_dict(data)


def _spec_from_dict(data: dict) -> DomainSpecFile:
    name = data.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        raise ValueError(
            f"domain spec: 'name' must match {_NAME_RE.pattern}, "
            f"got {name!r}"
        )
    board = data.get("board", "meta-arena")
    if board not in BOARDS:
        raise ValueError(
            f"domain spec {name!r}: board {board!r} unsupported "
            f"(v1 ships {sorted(BOARDS)!r})"
        )
    vocab = data.get("vocab")
    if vocab is not None and not isinstance(vocab, dict):
        raise ValueError(
            f"domain spec {name!r}: vocab must be an object"
        )
    vocab = vocab or {}
    ops = vocab.get("ops")
    attrs = vocab.get("attrs")
    handler_tags = vocab.get("handler_tags")
    rules_profile = _parse_rules(vocab.get("rules"))
    corpus = data.get("corpus")
    if corpus is not None and not isinstance(corpus, dict):
        raise ValueError(
            f"domain spec {name!r}: corpus must be an object"
        )
    corpus = corpus or {}
    kinds = corpus.get("kinds")
    player = data.get("player", "meta")
    if player not in PLAYERS:
        raise ValueError(
            f"domain spec {name!r}: player {player!r} unsupported "
            f"(v1 ships {sorted(PLAYERS)!r})"
        )
    incumbent = data.get("incumbent") or {}
    inc_kind = (
        incumbent.get("kind", "scripted")
        if isinstance(incumbent, dict)
        else incumbent
    )
    if inc_kind not in INCUMBENTS:
        raise ValueError(
            f"domain spec {name!r}: incumbent {inc_kind!r} "
            f"unsupported (v1 ships {sorted(INCUMBENTS)!r})"
        )
    shapes = data.get("shapes")
    hardware = data.get("hardware")
    if shapes is not None and not isinstance(shapes, dict):
        raise ValueError(
            f"domain spec {name!r}: shapes must be an object"
        )
    if hardware is not None and not isinstance(hardware, dict):
        raise ValueError(
            f"domain spec {name!r}: hardware must be an object"
        )
    shapes = shapes or {}
    hardware = hardware or {}
    tags = (
        _as_str_list(handler_tags, where=f"{name}.vocab.handler_tags")
        if handler_tags is not None
        else None
    )
    if tags is not None:
        unknown = sorted(set(tags) - set(lawdata.HANDLERS))
        if unknown:
            raise ValueError(
                f"domain spec {name!r}: unknown handler_tags {unknown!r} "
                f"(known: {sorted(lawdata.HANDLERS)!r})"
            )
    return DomainSpecFile(
        name=name,
        board=board,
        corpus_kinds=(
            _as_str_list(kinds, where=f"{name}.corpus.kinds")
            if kinds is not None
            else None
        ),
        corpus_coverage=bool(corpus.get("coverage", True)),
        vocab_ops=(
            frozenset(_as_str_list(ops, where=f"{name}.vocab.ops"))
            if ops is not None
            else None
        ),
        vocab_attrs=(
            frozenset(_as_str_list(attrs, where=f"{name}.vocab.attrs"))
            if attrs is not None
            else None
        ),
        handler_tags=tags,
        rules_profile=rules_profile,
        player=player,
        incumbent=inc_kind,
        shapes=dict(shapes),
        hardware=dict(hardware),
        description=str(data.get("description", "")),
        raw=data,
    )


# ---------------------------------------------------------------------------
#  Spec → domain
# ---------------------------------------------------------------------------


def _term_ops_attrs(term: Any) -> tuple[set, set]:
    """Collect ``(ops, attrs)`` over every ``Op`` subterm."""
    ops: set = set()
    attrs: set = set()
    stack = [term]
    seen: set[int] = set()
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, Op):
            ops.add(t.op)
            attrs.update((t.op, k) for k in t.attrs)
            stack.extend(t.args)
    return ops, attrs


def _ruleset_names(rules: Any) -> set:
    """Live rule names a case ruleset carries (``None`` → DEFAULT)."""
    if rules is None:
        return {r.name for r in DEFAULT}
    return {r.name for r in rules}


def _rule_allowlist(profile: Any) -> set | None:
    """Resolve a rules profile to allowed rule names (``None`` = all)."""
    if profile == "all":
        return None
    if profile == "default":
        return {r.name for r in DEFAULT}
    if profile == "scan":
        return _ruleset_names(corpus_cov.scan_rules())
    known = {r.name for r in corpus_cov.scan_rules()}
    unknown = sorted(set(profile) - known)
    if unknown:
        raise ValueError(f"vocab.rules: unknown rule names {unknown!r}")
    return set(profile)


def _case_ok(
    case: Any, spec: DomainSpecFile, allowed_rules: set | None
) -> bool:
    """Check one emitted case against the spec's vocab/rules bounds."""
    if allowed_rules is not None:
        names = _ruleset_names(case[2] if len(case) > 2 else None)
        if not names <= allowed_rules:
            return False
    if spec.vocab_ops is None and spec.vocab_attrs is None:
        return True
    ops, attrs = _term_ops_attrs(case[1])
    if spec.vocab_ops is not None and not ops <= spec.vocab_ops:
        return False
    if spec.vocab_attrs is not None:
        spelled = {f"{o}.{a}" for o, a in attrs}
        if not spelled <= spec.vocab_attrs:
            return False
    return True


def _kind_of(case: Any) -> str:
    return str(case[0]).split(":", 1)[-1]


def _eligible_kinds(spec: DomainSpecFile) -> list[str]:
    """Builder kinds whose sampled footprint fits the vocab.

    Samples ``_SAMPLE_CASES`` cases through the spec's own corpus
    selection and unions each kind's op/attr footprint; a kind is
    eligible iff its footprint is inside ``vocab.ops``/``vocab.attrs``
    (whichever are given) and its name survives ``corpus.kinds``.
    The per-case post-filter in :func:`_make_cases` enforces the
    same bounds on every emitted term — the footprint only decides
    what may *draw*.
    """
    sample = mp.gen_cases(
        0, _SAMPLE_CASES, coverage=spec.corpus_coverage
    )
    footprints: dict[str, tuple[set, set]] = {}
    for case in sample:
        kind = _kind_of(case)
        ops, attrs = _term_ops_attrs(case[1])
        prev_o, prev_a = footprints.get(kind, (set(), set()))
        footprints[kind] = (prev_o | ops, prev_a | attrs)
    out = []
    for kind, (ops, attrs) in sorted(footprints.items()):
        if (
            spec.corpus_kinds is not None
            and kind not in spec.corpus_kinds
        ):
            continue
        if spec.vocab_ops is not None and not ops <= spec.vocab_ops:
            continue
        if spec.vocab_attrs is not None:
            spelled = {f"{o}.{a}" for o, a in attrs}
            if not spelled <= spec.vocab_attrs:
                continue
        out.append(kind)
    if spec.corpus_kinds is not None:
        missing = sorted(set(spec.corpus_kinds) - set(footprints))
        if missing:
            raise ValueError(
                f"domain spec {spec.name!r}: corpus.kinds {missing!r} "
                "match no builder kind"
            )
    if not out:
        raise ValueError(
            f"domain spec {spec.name!r}: vocab/corpus selection "
            "leaves no eligible builder kinds"
        )
    return out


def _make_cases(spec: DomainSpecFile, kinds: list[str]) -> Any:
    """Return the ``(seed, n) -> [case, ...]`` generator.

    Draws through ``gen_cases`` restricted to the eligible kinds,
    then enforces the vocab/rules bounds per case — oversampling in
    rounds so post-filtered draws still fill *n* (or return short
    honestly when the eligible pool is thin).
    """
    allowed_rules = _rule_allowlist(spec.rules_profile)

    def cases(seed: int, n: int) -> list:
        out: list = []
        for rnd in range(_MAX_ROUNDS):
            if len(out) >= n:
                break
            chunk = mp.gen_cases(
                seed + rnd * 100_003,
                max(n, 24),
                coverage=spec.corpus_coverage,
                kinds=kinds,
            )
            for c in chunk:
                if _case_ok(c, spec, allowed_rules):
                    out.append(
                        (f"spec{len(out)}:{_kind_of(c)}", *c[1:])
                    )
                    if len(out) >= n:
                        break
        return out

    return cases


def _make_board(spec: DomainSpecFile) -> Any:
    """Return the ``case -> MetaArena`` factory with the handler subset."""
    if spec.handler_tags is None:
        handlers = None
    else:
        keep = set(spec.handler_tags)
        handlers = {
            tag: h for tag, h in lawdata.HANDLERS.items() if tag in keep
        }

    def board_of(case: Any) -> ma.MetaArena:
        rules = case[2] if len(case) > 2 else None
        sup = case[4] if len(case) > 4 else None
        return ma.MetaArena(
            case[1],
            rules,
            cost_fn=count_cost,
            supported=sup,
            handlers=handlers,
            max_nodes=20_000,
        )

    return board_of


# ---------------------------------------------------------------------------
#  Registries + the build
# ---------------------------------------------------------------------------

BOARDS: dict[str, Any] = {"meta-arena": _make_board}
PLAYERS: dict[str, Any] = {"meta": mp.META_SPEC}
INCUMBENTS: dict[str, Any] = {
    "scripted": ma.ScriptedPlayer,
    "greedy": ma.GreedyPlayer,
    "none": None,
}

#: Built spec records by name — the census/CLI reads these.
SPECS: dict[str, DomainSpecFile] = {}


def build_domain(spec: DomainSpecFile) -> Any:
    """Emit a :class:`~mlopt_discovery.play.Domain` for *spec*.

    Does not register — :func:`register_domain` does.  Lazily
    imports ``play`` so this module stays importable without the
    registry's full player-table cost.
    """
    from . import play

    kinds = _eligible_kinds(spec)
    scripted = INCUMBENTS[spec.incumbent]
    domain = play.spec_domain(
        _make_cases(spec, kinds),
        BOARDS[spec.board](spec),
        PLAYERS[spec.player],
        scripted=scripted,
    )
    SPECS[spec.name] = spec
    return domain


def register_domain(
    spec: DomainSpecFile, *, domains: dict | None = None
) -> Any:
    """Build *spec*'s domain and seat it in a ``DOMAINS`` registry.

    *domains* defaults to ``play.DOMAINS``; callers running
    ``play.py`` as ``__main__`` (``python -m mlopt_discovery.play``)
    must pass their own module's dict — ``__main__`` and the
    imported ``mlopt_discovery.play`` are distinct module objects
    with distinct registries.
    """
    from . import play

    domain = build_domain(spec)
    target = play.DOMAINS if domains is None else domains
    target[spec.name] = domain
    return domain


def register_file(
    source: str | Path | dict, *, domains: dict | None = None
) -> Any:
    """Load a spec (path / packaged name / mapping) and register it."""
    return register_domain(load_spec(source), domains=domains)


def spec_summary(spec: DomainSpecFile) -> dict:
    """Return the census a spec resolves to — kinds, vocab, bounds."""
    kinds = _eligible_kinds(spec)
    return {
        "name": spec.name,
        "board": spec.board,
        "player": spec.player,
        "incumbent": spec.incumbent,
        "corpus": {
            "coverage": spec.corpus_coverage,
            "kinds_requested": (
                list(spec.corpus_kinds)
                if spec.corpus_kinds is not None
                else None
            ),
            "kinds_eligible": kinds,
        },
        "vocab": {
            "ops": (
                len(spec.vocab_ops)
                if spec.vocab_ops is not None
                else None
            ),
            "attrs": (
                len(spec.vocab_attrs)
                if spec.vocab_attrs is not None
                else None
            ),
            "handler_tags": (
                list(spec.handler_tags)
                if spec.handler_tags is not None
                else None
            ),
            "rules": spec.rules_profile,
        },
        "recorded_only": {
            "shapes": spec.shapes,
            "hardware": spec.hardware,
        },
        "description": spec.description,
    }


# ---------------------------------------------------------------------------
#  CLI — spec census without episodes
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Print a spec's resolved census (no episodes played)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--spec",
        required=True,
        metavar="PATH|NAME",
        help="spec JSON path or packaged domains/<name>.json",
    )
    ap.add_argument(
        "--cases",
        type=int,
        default=12,
        help="sample cases to census through the real board path",
    )
    args = ap.parse_args(argv)
    spec = load_spec(args.spec)
    summary = spec_summary(spec)
    print(json.dumps(summary, indent=1, default=str))  # stdout-compat
    cases = _make_cases(spec, summary["corpus"]["kinds_eligible"])(
        0, args.cases
    )
    print(f"cases: {len(cases)}")  # stdout-compat
    for name, *_ in cases[: args.cases]:
        print(f"  {name}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
