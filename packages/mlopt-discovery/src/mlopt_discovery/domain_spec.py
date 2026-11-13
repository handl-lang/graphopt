"""Domain-spec files — Stage-2 self-driving, first cut: domains-as-data.

Stage 1 made the *corpus* mechanical (:mod:`corpus_cov` generates
cases from measured coverage gaps; ``overnight --auto-admit`` plus
eqsat's ``AUTO_TIER`` admit verified laws without a human).  Stage 2
makes the *domain* mechanical: a spec file names a vocabulary
subset, a corpus selection, a ruleset profile and an incumbent
recipe, and :func:`register_file` emits a whole playable
:class:`~mlopt_discovery.play.Domain` — cases generator, board
factory, player spec and scripted arm — into ``play.DOMAINS``.

Second-cut scope (honest):

* ``board`` — every shipped ``play`` domain is spec-able:
  ``"meta-arena"`` (the symbolic probe-tuple corpus on
  :class:`~mlopt_discovery.meta_arena.MetaArena` under
  ``count_cost``), ``"torch"`` (real ``nn.Module``s lifted
  through ``TorchSource`` and played under the sink's
  ``supported_ops`` bound, with the ``--deliver`` lowering hook),
  ``"gen"`` (the same lifted modules plus per-case *minted*
  genkernel handlers — ``corpus.builders`` selects the fusion
  builders, ``vocab.handler_tags`` subsets the ambient table),
  ``"joint"`` (forward+gradient programs on the meta-arena board),
  ``"search"`` (the ``SearchEnv`` rule-fire adapter —
  ``vocab.rules`` seats a rule-name subset, ``corpus.kinds`` the
  named case pool, ``corpus.horizon``/``patience`` the bounds),
  ``"fragment"`` (the ``vocab/`` fragment-application board —
  ``vocab.rules`` filters rewrite names, ``vocab.handler_tags``
  subsets the fragment handler table, ``corpus.max_moves`` the
  budget), ``"open"`` (the enumerated-candidate oracle game —
  ``corpus.ops``/``page``/``start``/``step`` are its only knobs),
  ``"player"`` (the spec-record rewrite meta-game —
  ``corpus.objective``/``max_evals``) and ``"bilinear"`` (the
  ``<2,2,2>`` decomposition board — ``corpus.scramble``).
  ``player``/``incumbent`` default per board and are validated
  against it — a spec cannot seat the meta-arena player spec on
  the search board.  Boards whose cases are not op-terms
  (``open``/``player``/``bilinear``) refuse ``vocab.*`` outright;
  boards that still genuinely need code (a new case generator, a
  new player vocabulary, a nested-objective scorer) are the
  plugin seam documented in the Stage-3 report — the registries
  below are the extension points.
* ``corpus.modules`` (torch board) — selects builders by name from
  ``play._torch_modules``'s zoo; ``corpus.kinds`` stays the
  meta-arena builder-kind seam.
* ``shapes`` / ``hardware`` — **consumed on the torch board**:
  ``shapes.widths``/``shapes.batch`` size the module draw and the
  example input, ``hardware.device``/``hardware.dtype`` place the
  model and input tensors.  On meta-arena they remain recorded
  only — the terms are symbolic, no tensor is materialized.
* ``vocab.rules`` — ``"all"`` (default), ``"default"``, ``"scan"``,
  an explicit name list, or ``{"base": ..., "minus": [names]}`` —
  the subtraction profile.
* ``player`` — ``"meta"`` or ``{"kind": "meta", "prior": <ref>}``;
  the prior re-seats the weight table (:func:`load_prior` — a
  packaged ``priors/<name>.json``, a path, an inline object, or
  ``"none"`` for the cold start).  :func:`bake_prior` mints a
  spec-matched table by oracle-imitation on the spec's own boards.
* ``vocab.ops`` / ``vocab.attrs`` — whitelists applied twice:
  per-kind *footprints* (sampled op/attr sets per builder kind)
  decide which kinds may draw at all on meta-arena, and a per-case
  post-filter enforces the invariant on every emitted term (a kind
  whose rare draws escape the footprint is dropped, not padded).
  On torch the lifted ``ir.root`` is an ordinary op-term, so the
  same post-filter applies per case.
* ``vocab.handler_tags`` — a subset of ``lawdata.HANDLERS`` seated
  on the board (``MetaArena(handlers=…)``); unlisted tags are
  never claimed.
* ``corpus.kinds`` / ``corpus.coverage`` — forwarded to
  :func:`meta_player.gen_cases`'s ``kinds``/``coverage`` seam; the
  effective draw is the intersection with the vocab-eligible set.
* ``incumbent`` — the scripted/baseline recipe: ``"scripted"``
  (``MetaArena.ScriptedPlayer`` — saturate→extract), ``"greedy"``
  (``GreedyPlayer`` — cheapest-immediate), ``"none"``.
* ``deliver`` — the optional delivery hook ``play --deliver``
  runs: ``"auto"`` (default — the board's own hook: ``torch``
  lowers+verifies the winning extraction through
  ``play._torch_deliver``), ``"none"`` (explicitly hookless), or
  ``"torch"`` (require the torch lowering — rejected at load on
  boards whose cases carry no example input).  ``true``/``false``
  spell ``"auto"``/``"none"``.  On ``meta-arena`` delivery is
  meaningless by construction — the cases are symbolic probe
  terms with no tensors, leaves or example input to lower and
  verify against — so ``"auto"`` resolves to no hook there and
  ``"torch"`` is refused.

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
    "DELIVERS",
    "INCUMBENTS",
    "PLAYERS",
    "SPECS",
    "DomainSpecFile",
    "bake_prior",
    "load_prior",
    "load_spec",
    "register_file",
    "spec_summary",
]

_DOMAINS_DIR = Path(__file__).resolve().parent / "domains"
_PRIORS_DIR = Path(__file__).resolve().parent / "priors"
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
    original mapping for fields the spec does not consume on the
    seated board (``shapes``/``hardware`` on meta-arena).
    """

    name: str
    board: str = "meta-arena"
    corpus_kinds: tuple[str, ...] | None = None
    corpus_modules: tuple[str, ...] | None = None
    corpus_builders: tuple[str, ...] | None = None
    corpus_ops: tuple[str, ...] | None = None
    corpus_objective: str | None = None
    corpus_params: dict = field(default_factory=dict)
    corpus_coverage: bool = True
    vocab_ops: frozenset[str] | None = None
    vocab_attrs: frozenset[str] | None = None  # "op.attr" spellings
    handler_tags: tuple[str, ...] | None = None
    rules_profile: Any = (
        "all"  # profile | ("minus", base, names)|tuple[name,...]
    )
    player: str = "meta"
    player_prior: Any = (
        None  # None | "none" | prior ref (name/path/dict)
    )
    incumbent: str = "scripted"
    deliver: str = "auto"  # "auto" | "none" | "torch"
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
    """Resolve ``vocab.rules`` to a profile, name tuple, or minus-form.

    Accepts ``"all"``/``"default"``/``"scan"``, an explicit name
    list, or ``{"base": <profile-or-names>, "minus": [names]}`` —
    the subtraction profile: the base allowlist with the named
    rules removed (``"all"`` resolves against the scan ruleset's
    name universe, the widest shipped set).
    """
    if value is None or value == "all":
        return "all"
    if value in ("default", "scan"):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, str) for v in value
    ):
        return tuple(value)
    if isinstance(value, dict):
        base = _parse_rules(value.get("base", "all"))
        minus = value.get("minus")
        if not isinstance(minus, (list, tuple)) or not all(
            isinstance(v, str) for v in minus
        ):
            raise ValueError(
                "vocab.rules.minus: expected a list of rule names, "
                f"got {minus!r}"
            )
        return ("minus", base, frozenset(minus))
    raise ValueError(
        f"vocab.rules: expected 'all'|'default'|'scan'|[names]|"
        f"{{'base','minus'}}, got {value!r}"
    )


# ---------------------------------------------------------------------------
#  Per-board field schemas — what a spec may name on each board
# ---------------------------------------------------------------------------
#
#  ``corpus`` keys each board consumes (anything else is a load
#  error — a spec that names a knob the board cannot read would
#  silently drift from what it claims).  ``vocab`` keys likewise.
#  ``_BOARD_DEFAULTS`` seats the player spec + incumbent playbook
#  a spec gets when it does not name them; ``_BOARD_INCUMBENTS``
#  bounds which playbook kinds may be seated explicitly (a
#  meta-arena ``ScriptedPlayer`` cannot read a ``SearchState``).

_CORPUS_KEYS: dict[str, frozenset] = {
    "meta-arena": frozenset({"kinds", "coverage"}),
    "torch": frozenset({"modules", "coverage"}),
    "gen": frozenset({"builders"}),
    "joint": frozenset({"kinds"}),
    "search": frozenset({"kinds", "horizon", "patience"}),
    "fragment": frozenset({"kinds", "max_moves"}),
    "open": frozenset({"ops", "page", "start", "step"}),
    "player": frozenset({"objective", "max_evals"}),
    "bilinear": frozenset({"scramble"}),
}

#: Integer ``corpus`` params per board (the rest are name lists or
#: the ``coverage`` flag parsed separately).
_CORPUS_PARAMS: dict[str, frozenset] = {
    "search": frozenset({"horizon", "patience"}),
    "fragment": frozenset({"max_moves"}),
    "open": frozenset({"page", "start", "step"}),
    "player": frozenset({"max_evals"}),
    "bilinear": frozenset({"scramble"}),
}

_VOCAB_KEYS: dict[str, frozenset] = {
    "meta-arena": frozenset({"ops", "attrs", "rules", "handler_tags"}),
    "torch": frozenset({"ops", "attrs", "rules", "handler_tags"}),
    "gen": frozenset({"ops", "attrs", "rules", "handler_tags"}),
    "joint": frozenset({"ops", "attrs", "rules", "handler_tags"}),
    "search": frozenset({"ops", "attrs", "rules"}),
    "fragment": frozenset({"ops", "attrs", "rules", "handler_tags"}),
    "open": frozenset(),
    "player": frozenset(),
    "bilinear": frozenset(),
}

_BOARD_DEFAULTS: dict[str, dict] = {
    "meta-arena": {"player": "meta", "incumbent": "scripted"},
    "torch": {"player": "meta", "incumbent": "scripted"},
    "gen": {"player": "meta", "incumbent": "scripted"},
    "joint": {"player": "meta", "incumbent": "scripted"},
    "search": {"player": "search", "incumbent": "search"},
    "fragment": {"player": "fragment", "incumbent": "fragment"},
    "open": {"player": "open", "incumbent": "none"},
    "player": {"player": "player", "incumbent": "player"},
    "bilinear": {"player": "bilinear", "incumbent": "bilinear"},
}

_BOARD_INCUMBENTS: dict[str, frozenset] = {
    "meta-arena": frozenset({"scripted", "greedy", "none"}),
    "torch": frozenset({"scripted", "greedy", "none"}),
    "gen": frozenset({"scripted", "greedy", "none"}),
    "joint": frozenset({"scripted", "greedy", "none"}),
    "search": frozenset({"search", "none"}),
    "fragment": frozenset({"fragment", "none"}),
    "open": frozenset({"none"}),
    "player": frozenset({"player", "none"}),
    "bilinear": frozenset({"bilinear", "none"}),
}

#: Boards whose cases carry an example input + the gen sink — the
#: ``deliver: "torch"`` lowering verifies against them (``gen``
#: cases keep the same ``ex`` record the torch cases do).
_TORCH_DELIVER_BOARDS: frozenset = frozenset({"torch", "gen"})

#: Boards ``bake_prior`` can imitate on — its oracle planner reads
#: meta-arena boards only (the others' states are not
#: ``MetaArena``-shaped).
_ORACLE_BOARDS: frozenset = frozenset(
    {"meta-arena", "torch", "gen", "joint"}
)

_REGISTRIES_READY = False
_EXTRA_CASES: dict[str, Any] = {}
_FRAGMENT_VOCAB: Any = None


def _ensure_registries() -> None:
    """Seat the ``play``-backed board/player/incumbent entries once.

    The play-module imports stay lazy so ``import
    mlopt_discovery.domain_spec`` keeps its light footprint; the
    first spec parse pays for them.  The registries are the plugin
    seam: a downstream board repo seats its own factories into
    ``BOARDS``/``_EXTRA_CASES``/``PLAYERS``/``INCUMBENTS`` before
    calling :func:`register_file` — the same shape kernelopt's
    ``mlopt_kernel.play.register()`` uses for ``play.DOMAINS``.
    """
    global _REGISTRIES_READY
    if _REGISTRIES_READY:
        return
    _REGISTRIES_READY = True
    from . import play as _p

    BOARDS.update(
        {
            "gen": _make_gen_board,
            "joint": _make_joint_board,
            "search": _make_search_board,
            "fragment": _make_fragment_board,
            "open": _make_open_board,
            "player": _make_player_board,
            "bilinear": _make_bilinear_board,
        }
    )
    PLAYERS.update(
        {
            "search": _p.SEARCH_SPEC,
            "fragment": _p.FRAGMENT_SPEC,
            "open": _p.og.OPEN_SPEC,
            "player": _p.pa.PLAYER_SPEC,
            "bilinear": _p.bg.BILINEAR_SPEC,
        }
    )
    INCUMBENTS.update(
        {
            "search": _p._scripted_search,
            "fragment": _p._fragment_scripted,
            "player": _p.pa.GreedyEditor,
            "bilinear": _p.bg.strassen_line,
        }
    )
    _EXTRA_CASES.update(
        {
            "gen": _make_gen_cases,
            "joint": _make_joint_cases,
            "search": _make_search_cases,
            "fragment": _make_fragment_cases,
            "open": _make_open_cases,
            "player": _make_player_cases,
            "bilinear": _make_bilinear_cases,
        }
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
    _ensure_registries()
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
    unknown_vocab = sorted(set(vocab) - _VOCAB_KEYS[board])
    if unknown_vocab:
        raise ValueError(
            f"domain spec {name!r}: board {board!r} has no "
            f"vocab.{unknown_vocab[0]} seam — its cases are not "
            f"op-terms the whitelist/post-filter can bind "
            f"(vocab keys for {board!r}: "
            f"{sorted(_VOCAB_KEYS[board]) or 'none'})"
        )
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
    unknown_corpus = sorted(set(corpus) - _CORPUS_KEYS[board])
    if unknown_corpus:
        raise ValueError(
            f"domain spec {name!r}: board {board!r} has no "
            f"corpus.{unknown_corpus[0]} seam (corpus keys for "
            f"{board!r}: {sorted(_CORPUS_KEYS[board])})"
        )
    kinds = corpus.get("kinds")
    modules = corpus.get("modules")
    builders = corpus.get("builders")
    corpus_ops = corpus.get("ops")
    objective = corpus.get("objective")
    if corpus_ops is not None:
        corpus_ops = _as_str_list(
            corpus_ops, where=f"{name}.corpus.ops"
        )
    if objective is not None and not isinstance(objective, str):
        raise ValueError(
            f"domain spec {name!r}: corpus.objective must be a "
            f"string, got {objective!r}"
        )
    params = {
        k: v
        for k, v in corpus.items()
        if k
        not in {
            "kinds",
            "modules",
            "builders",
            "ops",
            "objective",
            "coverage",
        }
    }
    bad_params = sorted(
        set(params) - _CORPUS_PARAMS.get(board, frozenset())
    )
    if bad_params:
        raise ValueError(
            f"domain spec {name!r}: board {board!r} has no "
            f"corpus.{bad_params[0]} param (int params for "
            f"{board!r}: {sorted(_CORPUS_PARAMS.get(board, ())) or 'none'})"
        )
    for k, v in params.items():
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(
                f"domain spec {name!r}: corpus.{k} must be an "
                f"int, got {v!r}"
            )
    if modules is not None and board != "torch":
        raise ValueError(
            f"domain spec {name!r}: corpus.modules selects torch "
            f"module-zoo builders — only board 'torch' has them "
            f"(board is {board!r})"
        )
    if builders is not None and board != "gen":
        raise ValueError(
            f"domain spec {name!r}: corpus.builders selects gen "
            f"fusion builders — only board 'gen' has them "
            f"(board is {board!r})"
        )
    if corpus_ops is not None and board != "open":
        raise ValueError(
            f"domain spec {name!r}: corpus.ops selects the open "
            f"board's candidate op vocabulary — only board 'open' "
            f"has it (board is {board!r}; vocab.ops is the term "
            "whitelist — a different field)"
        )
    if corpus_ops is not None:
        from mlopt_core.opmeta import REGISTRY as _OP_REGISTRY

        unknown_ops = sorted(set(corpus_ops) - set(_OP_REGISTRY))
        if unknown_ops:
            raise ValueError(
                f"domain spec {name!r}: corpus.ops names unknown "
                f"ops {unknown_ops!r} (opmeta registry is the "
                "vocabulary)"
            )
    if objective is not None and board != "player":
        raise ValueError(
            f"domain spec {name!r}: corpus.objective selects the "
            f"player board's nested objective — only board 'player' "
            f"has it (board is {board!r})"
        )
    if objective is not None:
        from . import play as _play

        if objective not in _play.pa.OBJECTIVES:
            raise ValueError(
                f"domain spec {name!r}: corpus.objective {objective!r} "
                f"unknown (known: {sorted(_play.pa.OBJECTIVES)!r})"
            )
    player_field = data.get("player")
    player_prior: Any = None
    if isinstance(player_field, dict):
        player = (
            player_field.get("kind") or _BOARD_DEFAULTS[board]["player"]
        )
        player_prior = player_field.get("prior")
    else:
        player = (
            player_field
            if player_field is not None
            else _BOARD_DEFAULTS[board]["player"]
        )
    if player not in PLAYERS:
        raise ValueError(
            f"domain spec {name!r}: player {player!r} unsupported "
            f"(v1 ships {sorted(PLAYERS)!r})"
        )
    if player != _BOARD_DEFAULTS[board]["player"]:
        raise ValueError(
            f"domain spec {name!r}: board {board!r} plays the "
            f"{_BOARD_DEFAULTS[board]['player']!r} player spec — "
            f"its legal-move/state shapes are bound to the board "
            f"(got player {player!r})"
        )
    if player_prior == "default":
        player_prior = None
    incumbent = data.get("incumbent")
    if incumbent is None or incumbent == {}:
        inc_kind = _BOARD_DEFAULTS[board]["incumbent"]
    else:
        if isinstance(incumbent, dict):
            inc_kind = (
                incumbent.get("kind")
                or _BOARD_DEFAULTS[board]["incumbent"]
            )
        else:
            inc_kind = incumbent
    if inc_kind not in INCUMBENTS:
        raise ValueError(
            f"domain spec {name!r}: incumbent {inc_kind!r} "
            f"unsupported (v1 ships {sorted(INCUMBENTS)!r})"
        )
    if inc_kind not in _BOARD_INCUMBENTS[board]:
        raise ValueError(
            f"domain spec {name!r}: board {board!r} cannot seat "
            f"incumbent {inc_kind!r} — its playbook reads a "
            f"different state shape (allowed: "
            f"{sorted(_BOARD_INCUMBENTS[board])})"
        )
    deliver = data.get("deliver", "auto")
    if deliver is True:
        deliver = "auto"
    elif deliver is False:
        deliver = "none"
    if deliver not in DELIVERS:
        raise ValueError(
            f"domain spec {name!r}: deliver {deliver!r} unsupported "
            f"(expected one of {sorted(DELIVERS)!r}, or a bool)"
        )
    if deliver == "torch" and board not in _TORCH_DELIVER_BOARDS:
        raise ValueError(
            f"domain spec {name!r}: deliver 'torch' lowers+verifies "
            f"against the case's example input — only boards "
            f"{sorted(_TORCH_DELIVER_BOARDS)!r} carry one (board is "
            f"{board!r}; meta-arena-family terms are symbolic, "
            "delivery is meaningless there)"
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
        known = _handler_tag_names(board)
        unknown = sorted(set(tags) - set(known))
        if unknown:
            raise ValueError(
                f"domain spec {name!r}: unknown handler_tags {unknown!r} "
                f"for board {board!r} (known: {sorted(known)!r})"
            )
    return DomainSpecFile(
        name=name,
        board=board,
        corpus_kinds=(
            _as_str_list(kinds, where=f"{name}.corpus.kinds")
            if kinds is not None
            else None
        ),
        corpus_modules=(
            _as_str_list(modules, where=f"{name}.corpus.modules")
            if modules is not None
            else None
        ),
        corpus_builders=(
            _as_str_list(builders, where=f"{name}.corpus.builders")
            if builders is not None
            else None
        ),
        corpus_ops=(
            _as_str_list(corpus_ops, where=f"{name}.corpus.ops")
            if corpus_ops is not None
            else None
        ),
        corpus_objective=(
            str(objective) if objective is not None else None
        ),
        corpus_params=dict(params),
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
        player_prior=player_prior,
        incumbent=inc_kind,
        deliver=deliver,
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
    """Resolve a rules profile to allowed rule names (``None`` = all).

    The ``("minus", base, names)`` form subtracts *names* from the
    base profile's allowlist; a ``"all"`` base resolves against the
    scan ruleset's name universe — the widest shipped set — so
    ``{"base": "all", "minus": [...]}`` is "everything but these".
    """
    if (
        isinstance(profile, tuple)
        and len(profile) == 3
        and profile[0] == "minus"
    ):
        _base, _minus = profile[1], set(profile[2])
        base = _rule_allowlist(_base)
        if base is None:
            base = _ruleset_names(corpus_cov.scan_rules())
        return base - _minus
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
    handlers = _handler_subset(spec)

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


def _handler_subset(spec: DomainSpecFile) -> dict | None:
    """Resolve ``vocab.handler_tags`` to the seated handler table."""
    if spec.handler_tags is None:
        return None
    keep = set(spec.handler_tags)
    return {
        tag: h for tag, h in lawdata.HANDLERS.items() if tag in keep
    }


def _torch_widths(shapes: dict) -> list[int]:
    """Resolve ``shapes.widths`` — explicit positive-int list."""
    widths = shapes.get("widths")
    if widths is None:
        return [4, 8, 16]
    if (
        not isinstance(widths, list)
        or not widths
        or any(not isinstance(w, int) or w <= 0 for w in widths)
    ):
        raise ValueError(
            f"shapes.widths: expected a non-empty list of positive "
            f"ints, got {widths!r}"
        )
    return list(widths)


def _torch_batch(shapes: dict) -> int:
    """Resolve ``shapes.batch`` — the example input's leading dim."""
    batch = shapes.get("batch", 4)
    if not isinstance(batch, int) or batch <= 0:
        raise ValueError(
            f"shapes.batch: expected a positive int, got {batch!r}"
        )
    return batch


def _torch_device(hardware: dict) -> str | None:
    device = hardware.get("device")
    if device is None:
        return None
    if not isinstance(device, str):
        raise ValueError(
            f"hardware.device: expected a device string, got {device!r}"
        )
    return device


def _make_torch_cases(spec: DomainSpecFile) -> Any:
    """Return the ``(seed, n) -> [case, ...]`` torch-module stream.

    Draws through ``play._torch_cases`` — the same lifted-module
    cases the built-in ``torch`` domain plays — restricted to the
    spec's ``corpus.modules`` selection, ``shapes.widths``/``batch``
    and ``hardware.device``/``dtype``, then enforces the vocab
    bounds per case (the lifted ``ir.root`` is an ordinary op-term,
    so ``vocab.ops``/``vocab.attrs`` apply unchanged).  Torch
    imports lazily inside the generator.
    """
    from . import play as _play

    widths = _torch_widths(spec.shapes)
    batch = _torch_batch(spec.shapes)
    device = _torch_device(spec.hardware)
    dtype = spec.hardware.get("dtype")
    allowed_rules = _rule_allowlist(spec.rules_profile)
    if allowed_rules is not None:
        # torch cases carry ``rules=None`` → the DEFAULT nameset;
        # a profile that excludes any of them leaves an empty stream
        default_names = {r.name for r in DEFAULT}
        if not default_names <= allowed_rules:
            raise ValueError(
                f"domain spec {spec.name!r}: board 'torch' cases "
                "run the DEFAULT ruleset — vocab.rules admits "
                "none of its names"
            )

    def cases(seed: int, n: int) -> list:
        out: list = []
        for rnd in range(_MAX_ROUNDS):
            if len(out) >= n:
                break
            chunk = _play._torch_cases(
                seed + rnd * 100_003,
                max(n, 24),
                modules=spec.corpus_modules,
                widths=widths,
                batch=batch,
                device=device,
                dtype=dtype,
            )
            for c in chunk:
                if _case_ok(c, spec, allowed_rules):
                    out.append(c)
                    if len(out) >= n:
                        break
        return out

    return cases


def _make_torch_board(spec: DomainSpecFile) -> Any:
    """Torch-lifted case → MetaArena under the sink bound + subset."""
    handlers = _handler_subset(spec)

    def board_of(case: Any) -> ma.MetaArena:
        rules = case[2] if len(case) > 2 else None
        return ma.MetaArena(
            case[1],
            rules,
            cost_fn=count_cost,
            supported=case[4] if len(case) > 4 else None,
            handlers=handlers,
            max_nodes=20_000,
        )

    return board_of


# ---------------------------------------------------------------------------
#  The code-shaped boards — gen / joint / search / fragment / open /
#  player / bilinear.  Each wires the shipped ``play`` generator and
#  board factory; the spec's corpus/vocab fields are the only data
#  knobs (see ``_CORPUS_KEYS``/``_VOCAB_KEYS``).
# ---------------------------------------------------------------------------


def _fragment_vocab() -> Any:
    """Return the loaded ``vocab/`` directory (cached)."""
    global _FRAGMENT_VOCAB
    if _FRAGMENT_VOCAB is None:
        from . import vocabdir

        directory = vocabdir.default_vocab_dir()
        if directory is None:
            raise ValueError(
                "no vocab directory — the fragment board needs "
                "vocab/ or MLOPT_VOCAB_DIR"
            )
        _FRAGMENT_VOCAB = vocabdir.load_vocab(directory)
    return _FRAGMENT_VOCAB


def _handler_tag_names(board: str) -> Any:
    """Return the handler-tag universe *board* validates against.

    Most boards seat subsets of ``lawdata.HANDLERS``; the fragment
    board's table is the ``vocab/`` directory's own
    ``handler_table()`` — a different (larger) namespace.
    """
    if board == "fragment":
        return _fragment_vocab().handler_table()
    return lawdata.HANDLERS


def _fragment_handler_subset(spec: DomainSpecFile) -> dict | None:
    """Resolve ``vocab.handler_tags`` against the fragment table."""
    if spec.handler_tags is None:
        return None
    table = _fragment_vocab().handler_table()
    keep = set(spec.handler_tags)
    return {tag: table[tag] for tag in keep if tag in table}


def _default_rules_filtered(
    spec: DomainSpecFile, ruleset: Any, *, board: str
) -> Any:
    """Return *ruleset* filtered to the spec's rules allowlist.

    ``None`` (profile ``"all"``) seats the board's own default —
    the factory applies it.  An allowlist that survives no rule
    name refuses at build: an empty ruleset is a spec bug, not a
    domain.
    """
    allowed = _rule_allowlist(spec.rules_profile)
    if allowed is None:
        return None
    rules = tuple(r for r in ruleset if r.name in allowed)
    if not rules:
        raise ValueError(
            f"domain spec {spec.name!r}: vocab.rules leaves no "
            f"rules for board {board!r}"
        )
    return rules


def _collect(
    out: list,
    chunk: Any,
    spec: DomainSpecFile,
    n: int,
) -> None:
    """Append vocab-passing cases, renamed ``spec<i>:<kind>``."""
    for c in chunk:
        if len(out) >= n:
            return
        if _case_ok(c, spec, None):
            out.append((f"spec{len(out)}:{_kind_of(c)}", *c[1:]))


def _make_gen_cases(spec: DomainSpecFile) -> Any:
    """Gen-kernel case stream — ``corpus.builders`` + ambient subset."""
    from . import play as _play

    builders = spec.corpus_builders
    ambient = _handler_subset(spec)
    # eager check: ``_gen_cases`` validates builder names before its
    # draw loop — an ``n=0`` probe refuses unknown names at build,
    # not on the first episode (it does import torch once here).
    _play._gen_cases(0, 0, builders=builders, ambient_handlers=ambient)
    allowed = _rule_allowlist(spec.rules_profile)
    if allowed is not None:
        # the gen board always seats DEFAULT + the minted store —
        # a profile must admit every DEFAULT name (same contract
        # as the torch board's rules=None cases)
        default_names = {r.name for r in DEFAULT}
        if not default_names <= allowed:
            raise ValueError(
                f"domain spec {spec.name!r}: board 'gen' seats the "
                "DEFAULT ruleset plus minted handlers — vocab.rules "
                "admits none of its names"
            )

    def cases(seed: int, n: int) -> list:
        out: list = []
        for rnd in range(_MAX_ROUNDS):
            if len(out) >= n:
                break
            chunk = _play._gen_cases(
                seed + rnd * 100_003,
                max(n, 24),
                builders=builders,
                ambient_handlers=ambient,
            )
            _collect(out, chunk, spec, n)
        return out

    return cases


def _make_gen_board(spec: DomainSpecFile) -> Any:
    """Gen case → MetaArena under its minted handler store.

    The ambient-handler subset rides inside each case's ``ex``
    record (``_gen_cases`` seats it); ``_gen_board`` reads it —
    nothing spec-side remains to bind here.
    """
    from . import play as _play

    return _play._gen_board


def _make_joint_cases(spec: DomainSpecFile) -> Any:
    """Joint-program stream — ``corpus.kinds`` picks archetypes."""
    from . import play as _play

    kinds = spec.corpus_kinds
    if kinds is not None:
        unknown = sorted(set(kinds) - {"0", "1", "2", "3"})
        if unknown:
            raise ValueError(
                f"domain spec {spec.name!r}: corpus.kinds {unknown!r} "
                "match no joint archetype (known: '0'..'3')"
            )

    def cases(seed: int, n: int) -> list:
        out: list = []
        for rnd in range(_MAX_ROUNDS):
            if len(out) >= n:
                break
            chunk = _play._joint_cases(seed + rnd * 100_003, max(n, 24))
            for c in chunk:
                if kinds is not None and _kind_of(c) not in kinds:
                    continue
                _collect(out, [c], spec, n)
        return out

    return cases


def _make_joint_board(spec: DomainSpecFile) -> Any:
    """Joint term → MetaArena under the profiled rules + subset."""
    handlers = _handler_subset(spec)
    rules = _default_rules_filtered(spec, DEFAULT, board="joint")

    def board_of(case: Any) -> ma.MetaArena:
        return ma.MetaArena(
            case[1],
            rules,
            cost_fn=count_cost,
            handlers=handlers,
            max_nodes=20_000,
        )

    return board_of


def _make_search_cases(spec: DomainSpecFile) -> Any:
    """SearchEnv case stream — ``corpus.kinds`` picks pool names."""
    from . import play as _play

    kinds = spec.corpus_kinds

    def cases(seed: int, n: int) -> list:
        out: list = []
        for rnd in range(_MAX_ROUNDS):
            if len(out) >= n:
                break
            chunk = _play._search_cases(
                seed + rnd * 100_003, max(n, 24), kinds=kinds
            )
            _collect(out, chunk, spec, n)
        return out

    return cases


def _make_search_board(spec: DomainSpecFile) -> Any:
    """Search case → SearchBoard under the profiled ruleset."""
    from . import play as _play

    rules = _default_rules_filtered(spec, DEFAULT, board="search")
    params = spec.corpus_params

    def board_of(case: Any) -> Any:
        return _play._search_board(
            case,
            rules,
            horizon=params.get("horizon", 12),
            patience=params.get("patience"),
        )

    return board_of


def _make_fragment_cases(spec: DomainSpecFile) -> Any:
    """Fragment-move case stream — ``corpus.kinds`` picks pool names."""
    from . import play as _play

    kinds = spec.corpus_kinds

    def cases(seed: int, n: int) -> list:
        out: list = []
        for rnd in range(_MAX_ROUNDS):
            if len(out) >= n:
                break
            chunk = _play._fragment_cases(
                seed + rnd * 100_003, max(n, 24), kinds=kinds
            )
            _collect(out, chunk, spec, n)
        return out

    return cases


def _make_fragment_board(spec: DomainSpecFile) -> Any:
    """Fragment case → FragmentArena under profiled vocab subsets."""
    from . import play as _play

    vocab = _fragment_vocab()
    laws, prims, schemas = _play.fg._rewrite_kinds(vocab)
    rewrites = {**laws, **prims, **schemas}
    rules = _default_rules_filtered(
        spec, rewrites.values(), board="fragment"
    )
    handlers = _fragment_handler_subset(spec)
    max_moves = spec.corpus_params.get("max_moves", 256)

    def board_of(case: Any) -> Any:
        return _play._fragment_board(
            case, rules=rules, handlers=handlers, max_moves=max_moves
        )

    return board_of


def _make_open_cases(spec: DomainSpecFile) -> Any:
    """Open-game page stream — ``corpus.ops``/``page``/``start``/``step``."""
    from . import open_game as og

    ops = spec.corpus_ops
    params = spec.corpus_params

    def cases(seed: int, n: int) -> list:
        return og.gen_cases(
            seed,
            n,
            ops=ops,
            page=params.get("page", 48),
            start=params.get("start", 0),
            step=params.get("step", 1),
        )

    return cases


def _make_open_board(spec: DomainSpecFile) -> Any:
    """Open case dict → OpenArena (the case *is* the board spec)."""
    from . import open_game as og

    return og.OpenArena


def _make_player_cases(spec: DomainSpecFile) -> Any:
    """Player-arena case stream — ``corpus.objective``/``max_evals``."""
    from . import player_arena as pa

    objective = spec.corpus_objective or "open"
    kwargs: dict[str, Any] = {"objective": objective}
    if "max_evals" in spec.corpus_params:
        kwargs["max_evals"] = spec.corpus_params["max_evals"]

    def cases(seed: int, n: int) -> list:
        return pa.gen_cases(seed, n, **kwargs)

    return cases


def _make_player_board(spec: DomainSpecFile) -> Any:
    """Player case dict → PlayerArena."""
    from . import player_arena as pa

    return pa.PlayerArena


def _make_bilinear_cases(spec: DomainSpecFile) -> Any:
    """Bilinear case stream — ``corpus.scramble`` sizes the draw."""
    from . import bilinear_game as bg

    scramble = spec.corpus_params.get("scramble", 8)

    def cases(seed: int, n: int) -> list:
        return bg.gen_cases(seed, n, scramble=scramble)

    return cases


def _make_bilinear_board(spec: DomainSpecFile) -> Any:
    """Bilinear case → BilinearBoard."""
    from . import bilinear_game as bg

    return bg.BilinearBoard


# ---------------------------------------------------------------------------
#  Player priors — spec-matched weight tables
# ---------------------------------------------------------------------------


def load_prior(source: Any) -> dict[str, float]:
    """Load a weight table ``{feature: logit-weight}``.

    *source* is a packaged prior name (``priors/<name>.json`` inside
    the package), a filesystem path to such a JSON object, or an
    inline mapping.  ``"none"`` resolves to the cold-start empty
    table.  Tables come out of :func:`bake_prior` (or a hand edit) —
    keys are feature names in the player spec's schema, values the
    logit weights; missing keys read as ``0``.
    """
    if source == "none":
        return {}
    if isinstance(source, dict):
        data = dict(source)
    else:
        if not isinstance(source, str):
            raise ValueError(
                f"player.prior: expected a name, path, or object, "
                f"got {source!r}"
            )
        path = Path(source)
        if not path.exists() and not source.endswith(".json"):
            cand = _PRIORS_DIR / f"{source}.json"
            if cand.exists():
                path = cand
        if not path.exists():
            raise ValueError(f"player prior not found: {source!r}")
        data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("player prior: top level must be an object")
    out: dict[str, float] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, (int, float)):
            raise ValueError(
                f"player prior: entries must be str -> number, "
                f"got {k!r}: {v!r}"
            )
        out[k] = float(v)
    return out


def _resolve_player(spec: DomainSpecFile) -> Any:
    """Return the domain's player :class:`DomainSpec`.

    The base spec (``PLAYERS[spec.player]``) carries the shipped
    prior; ``player.prior`` re-seats it — a packaged/path/inline
    weight table via :func:`load_prior`, ``"none"`` for the uniform
    cold start, or absent/``"default"`` for the shipped bake.  The
    re-seat rebuilds the spec over the same parts/legal/features —
    only the weight table changes.
    """
    from gamekit.players import DomainSpec

    base = PLAYERS[spec.player]
    if spec.player_prior is None:
        return base
    weights = load_prior(spec.player_prior)
    return DomainSpec.build(
        features=base.features,
        legal=base.legal,
        parts=base.parts,
        weights=weights,
        rank=base.rank,
        lr=base.lr,
    )


def bake_prior(
    spec: DomainSpecFile | Any,
    *,
    cases: int = 120,
    seed: int = 0,
    budget: int = 12,
    imitate_epochs: int = 2,
    rl_epochs: int = 0,
    init: str = "spec",
) -> dict[str, float]:
    """Bake a spec-matched player prior — oracle-imitation (+RL).

    Plans the meta-arena oracle line per case *on the spec domain's
    own boards* (handler subset and supported bound included — the
    demonstrations can only name moves the spec board offers), then
    replays them through ``imitate_oracle`` for *imitate_epochs*
    passes, optionally followed by *rl_epochs* REINFORCE passes via
    ``gamekit.engine.train_policy``.  *init* ``"spec"`` warm-starts
    from the resolved spec weights (the shipped bake's recipe —
    fine-tune the prior on the narrowed corpus); ``"cold"`` starts
    from ``{}``.  Returns the ``{feature: weight}`` table
    :func:`load_prior` reads — round to 6 decimals, zeros dropped.
    """
    if init not in ("spec", "cold"):
        raise ValueError(
            f"bake_prior: init must be 'spec'|'cold', got {init!r}"
        )
    from gamekit import engine as eng

    spec_obj: DomainSpecFile | None = None
    if hasattr(spec, "board_of"):
        domain: Any = spec
    else:
        spec_obj = spec
        if spec_obj.board not in _ORACLE_BOARDS:
            raise ValueError(
                f"bake_prior: board {spec_obj.board!r} states are "
                "not MetaArena-shaped — the oracle planner cannot "
                f"imitate on it (oracle boards: "
                f"{sorted(_ORACLE_BOARDS)!r})"
            )
        domain = build_domain(spec)
    case_list = list(domain.cases(seed, cases))
    plans = mp.plan_oracles(
        case_list,
        budget=budget,
        handlers=(
            _handler_subset(spec_obj) if spec_obj is not None else None
        ),
        board_of=domain.board_of,
    )
    player = mp.MetaLearnedPlayer(
        seed=seed,
        spec=domain.spec,
        weights=(
            {} if init == "cold" else dict(domain.spec.weights or {})
        ),
        temperature=1.0,
    )
    for _ in range(imitate_epochs):
        mp.imitate_oracle(
            player,
            case_list,
            plans,
            budget=budget,
            board_of=domain.board_of,
        )
    for _ in range(rl_epochs):
        eng.train_policy(
            player, case_list, domain.board_of, budget=budget
        )
    return {
        k: round(v, 6)
        for k, v in player.weights_dict().items()
        if round(v, 6) != 0.0
    }


# ---------------------------------------------------------------------------
#  Registries + the build
# ---------------------------------------------------------------------------

BOARDS: dict[str, Any] = {
    "meta-arena": _make_board,
    "torch": _make_torch_board,
}
PLAYERS: dict[str, Any] = {"meta": mp.META_SPEC}
INCUMBENTS: dict[str, Any] = {
    "scripted": ma.ScriptedPlayer,
    "greedy": ma.GreedyPlayer,
    "none": None,
}
#: ``deliver`` spellings a spec may name — ``"auto"`` defers to the
#: board's own hook (see :func:`_deliver_for`), ``"none"`` seats no
#: hook even where the board has one, ``"torch"`` requires the
#: torch lowering (load-validated to board ``"torch"``).
DELIVERS: dict[str, Any] = {"auto": None, "none": None, "torch": None}


def _deliver_for(spec: DomainSpecFile) -> Any:
    """Return the spec board's deliver hook (``None`` when absent).

    ``spec.deliver`` gates first: ``"none"`` always seats nothing,
    ``"torch"`` seats ``play._torch_deliver`` (load validation
    already pinned the board), ``"auto"`` defers to the board —
    torch delivers, the gen board delivers through its gen sink,
    meta-arena-family symbolic boards do not (no lowering or
    verify target).
    """
    if spec.deliver == "none":
        return None
    if spec.deliver == "torch" or (
        spec.deliver == "auto" and spec.board in _TORCH_DELIVER_BOARDS
    ):
        from . import play as _play

        return _play._torch_deliver
    return None


#: Built spec records by name — the census/CLI reads these.
SPECS: dict[str, DomainSpecFile] = {}


def build_domain(spec: DomainSpecFile) -> Any:
    """Emit a :class:`~mlopt_discovery.play.Domain` for *spec*.

    Does not register — :func:`register_domain` does.  Lazily
    imports ``play`` so this module stays importable without the
    registry's full player-table cost.
    """
    _ensure_registries()
    from . import play

    if spec.board == "meta-arena":
        kinds = _eligible_kinds(spec)
        cases_fn = _make_cases(spec, kinds)
    elif spec.board == "torch":
        cases_fn = _make_torch_cases(spec)
    elif spec.board in _EXTRA_CASES:
        cases_fn = _EXTRA_CASES[spec.board](spec)
    else:  # pragma: no cover - BOARDS validation gates this
        raise ValueError(f"unsupported board {spec.board!r}")
    scripted = INCUMBENTS[spec.incumbent]
    domain = play.spec_domain(
        cases_fn,
        BOARDS[spec.board](spec),
        _resolve_player(spec),
        scripted=scripted,
        deliver=_deliver_for(spec),
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
    corpus: dict[str, Any] = {
        "coverage": spec.corpus_coverage,
        "kinds_requested": (
            list(spec.corpus_kinds)
            if spec.corpus_kinds is not None
            else None
        ),
        "modules": (
            list(spec.corpus_modules)
            if spec.corpus_modules is not None
            else None
        ),
        "builders": (
            list(spec.corpus_builders)
            if spec.corpus_builders is not None
            else None
        ),
        "ops": (
            list(spec.corpus_ops)
            if spec.corpus_ops is not None
            else None
        ),
        "objective": spec.corpus_objective,
        "params": spec.corpus_params or None,
    }
    if spec.board == "meta-arena":
        corpus["kinds_eligible"] = _eligible_kinds(spec)
    return {
        "name": spec.name,
        "board": spec.board,
        "player": spec.player,
        "player_prior": spec.player_prior,
        "incumbent": spec.incumbent,
        "deliver": spec.deliver,
        "corpus": corpus,
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
        "shapes": spec.shapes,
        "hardware": spec.hardware,
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
    if spec.board == "meta-arena":
        cases_fn = _make_cases(
            spec, summary["corpus"]["kinds_eligible"]
        )
    elif spec.board == "torch":
        cases_fn = _make_torch_cases(spec)
    else:
        cases_fn = _EXTRA_CASES[spec.board](spec)
    cases = cases_fn(0, args.cases)
    print(f"cases: {len(cases)}")  # stdout-compat
    for c in cases[: args.cases]:
        label = c.get("name") if isinstance(c, dict) else c[0]
        print(f"  {label}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
