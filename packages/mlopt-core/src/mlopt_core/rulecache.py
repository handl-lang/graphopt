"""Persistent cache for synthesized rewrite rules.

:func:`mlopt_core.meta.synthesize_rules` is the expensive phase of the
meta-optimization loop — critical-pair completion plus per-candidate
validation — but its output is pure data.  This module persists it:
``synthesize_rules_cached`` returns the same derived rules on a second
run without re-deriving them.

Serialization scheme
--------------------
A synthesized :class:`Rewrite` has Op-tree lhs/rhs patterns (metavar
strings, ``$attr:`` attribute metavars, ``@1:``/``@2:`` derive
placeholders) plus Python callables for ``check``/``derive``.  The
callables are NOT serialized — instead each guarded rule stores its
*re-expression spec*: the parent pair names and the two substitution
maps ``(pat1, pat2)`` that map each parent's metavariables to patterns
over the derived rule's metavars (recorded on ``rule.guard_pats`` at
synthesis time).  At load time the composite hooks are rebuilt by
re-running :func:`meta._compose_guards` against the parent rules
resolved by name from the caller's ruleset — the composition machinery
is cheap; it is the search/validation that was expensive.

Cache key
---------
``sha256`` over ``{version, ruleset_fp, seeds_fp, params}`` where

* ``ruleset_fp`` — hash of the sorted per-rule records
  ``{name, lhs, rhs, check_sig, derive_sig}`` (hook signatures hash
  source text when inspectable, else the qualified name — worst case a
  stale signature just misses);
* ``seeds_fp`` — hash of the serialized seed terms, in order;
* ``params`` — fuel / numeric_check / require_overlap / emit_subsumed
  plus the fixpoint knobs: ``rounds`` / ``max_emitted`` (both change
  the mint, so both key) and ``observer`` — the ``on_round`` callable
  enters the key through its :func:`_hook_sig` proxy, never the object
  itself: a vetoing observer truncates the fixpoint and must not share
  the unobserved entry; an uninspectable signature degrades to a
  miss.  (A *pure* observer — one that only records strata — changes
  the key too; that is the conservative direction.)

Any change to the parent ruleset, seeds, or synthesis parameters
produces a different key — a stale cache misses cleanly rather than
silently loading rules derived under different assumptions.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import inspect
import json
import os
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import mlopt_core.meta as M
from mlopt_core.egraph import Rewrite
from mlopt_core.ir import (
    attr_from_data,
    attr_to_data,
    term_from_data,
    term_to_data,
)
from mlopt_core.laws import cond

__all__ = [
    "RuleCache",
    "cache_key",
    "ruleset_fingerprint",
    "seed_fingerprint",
    "synthesize_rules_cached",
]

#: Bump when the on-disk format or reconstruction semantics change.
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
#  Term / attribute serialization  (Op trees, metvars, placeholders — pure data)
#
#  The codec itself lives at the IR layer — ``mlopt_core.ir.term_to_data``
#  and friends — where ``laws.serialize`` (the lemma-store seam) reaches it
#  without pulling in ``meta``.  The historical private names stay as
#  aliases: the wire format is unchanged, so existing caches and
#  fingerprints hold.
# ---------------------------------------------------------------------------

_enc_attr = attr_to_data
_dec_attr = attr_from_data
_enc_term = term_to_data
_dec_term = term_from_data


def _enc_binding(m: dict) -> list:
    """Encode a substitution/re-expression map.

    ``$attr:`` values are attribute values (concrete constants or
    metavar-name strings); every other key maps to a term/pattern.
    """
    out = []
    for k, v in m.items():
        if k.startswith("$attr:"):
            out.append([k, {"a": _enc_attr(v)}])
        else:
            out.append([k, {"t": _enc_term(v)}])
    return out


def _dec_binding(pairs: list) -> dict:
    out = {}
    for k, enc in pairs:
        if "a" in enc:
            out[k] = _dec_attr(enc["a"])
        else:
            out[k] = _dec_term(enc["t"])
    return out


# ---------------------------------------------------------------------------
#  Rule records
# ---------------------------------------------------------------------------


def _enc_rule(rule: Rewrite) -> dict:
    """Serialize a synthesized rule.

    Records name, law, provenance, patterns, and — when guarded — the
    (pat1, pat2) re-expression spec recorded on ``rule.guard_pats``.
    Callables are never serialized.
    """
    spec = getattr(rule, "guard_pats", None)
    guarded = rule.check is not None or rule.derive is not None
    if guarded and spec is None:
        raise TypeError(
            f"rule {rule.name!r} has check/derive hooks but no "
            "guard_pats re-expression spec — only rules emitted by "
            "meta.synthesize_rules are cacheable"
        )
    return {
        "name": rule.name,
        "law": rule.law,
        "parents": list(
            getattr(rule, "parents", M.SYNTH_PARENTS.get(rule.name, ()))
        ),
        "lhs": _enc_term(rule.lhs),
        "rhs": _enc_term(rule.rhs),
        "guard": (
            {
                "pat1": _enc_binding(spec[0]),
                "pat2": _enc_binding(spec[1]),
            }
            if (guarded and spec is not None)
            else None
        ),
    }


def _dec_rule(
    rec: dict, parents_by_name: dict[str, Rewrite]
) -> Rewrite:
    """Rebuild a rule.

    Guarded rules re-run ``_compose_guards`` on the stored
    re-expression maps — the parents must be present (by name) in
    *parents_by_name*, or a ``KeyError`` propagates and the whole load
    is treated as a miss.
    """
    lhs = _dec_term(rec["lhs"])
    rhs = _dec_term(rec["rhs"])
    check = derive = None
    pats = None
    guard = rec.get("guard")
    if guard is not None:
        p1, p2 = rec["parents"]
        r1 = parents_by_name[p1]
        r2 = parents_by_name[p2]
        pats = (
            _dec_binding(guard["pat1"]),
            _dec_binding(guard["pat2"]),
        )
        check, drv = M._compose_guards(r1, r2, pats[0], pats[1])
        # Same policy as synthesize_rules.offer: the composite derive is
        # only attached when the RHS actually carries "@i:" placeholders.
        if not (drv is not None and M._rhs_derive_placeholders(rhs)):
            drv = None
        derive = drv
    rule = Rewrite(
        name=rec["name"],
        lhs=lhs,
        rhs=rhs,
        law=rec["law"],
        check=check,
        derive=derive,
    )
    parents = tuple(rec.get("parents") or ())
    object.__setattr__(rule, "parents", parents)
    if parents:
        M.SYNTH_PARENTS[rule.name] = parents
    if pats is not None:
        object.__setattr__(rule, "guard_pats", pats)
    return rule


# ---------------------------------------------------------------------------
#  Fingerprints / cache key
# ---------------------------------------------------------------------------


def _sha(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()


def _hook_sig(fn) -> str:
    """Stability signature for a check/derive callable.

    Source text when inspectable, else module+qualname.  An unstable
    signature only ever causes a cache MISS, never a wrong hit —
    conservative by design.
    """
    if fn is None:
        return ""
    if isinstance(fn, functools.partial):
        return (
            f"partial({_hook_sig(fn.func)};{fn.args!r};{fn.keywords!r})"
        )
    ident = f"{getattr(fn, '__module__', '')}:{getattr(fn, '__qualname__', '')}"
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = ""
    return hashlib.sha256(f"{ident}|{src}".encode()).hexdigest()[:16]


def ruleset_fingerprint(rules: Iterable[Rewrite]) -> str:
    """Hash of the parent ruleset.

    Covers every rule's name, serialized lhs/rhs patterns, and hook
    signatures — sorted so rule order doesn't matter.
    """
    entries = sorted(
        json.dumps(
            {
                "name": r.name,
                "lhs": _enc_term(r.lhs),
                "rhs": _enc_term(r.rhs),
                "check": _hook_sig(r.check),
                "derive": _hook_sig(r.derive),
                # Declarative conditions are data: a folded ``cond``
                # shares one ``compile_guard`` closure signature
                # across every cond-carrying rule, so the hook sig
                # cannot see a cond edit — hash the data itself.
                "cond": cond.cond_to_data(getattr(r, "cond", None)),
                # Same story for a folded ``dspec`` — one
                # ``compile_derive`` closure signature across every
                # spec-carrying rule.
                "dspec": cond.derive_to_data(getattr(r, "dspec", None)),
            },
            sort_keys=True,
        )
        for r in rules
    )
    return _sha(entries)


def seed_fingerprint(seed_terms: Iterable[Any]) -> str:
    """Hash of the serialized seed terms, in order.

    Seed order affects which derivations are explored and rule naming.
    """
    return _sha(
        json.dumps(_enc_term(s), sort_keys=True) for s in seed_terms
    )


def cache_key(
    rules: Iterable[Rewrite],
    seed_terms: Iterable[Any] = (),
    *,
    fuel: int = 512,
    numeric_check: bool = True,
    require_overlap: bool = True,
    emit_subsumed: bool = False,
    rounds: int = 1,
    max_emitted: int | None = None,
    on_round: Callable[[int, list[Rewrite]], object] | None = None,
) -> str:
    """Return the cache key for one synthesis problem instance.

    Combines the ruleset fingerprint, seed fingerprint, synthesis
    params, and format version.  ``rounds`` / ``max_emitted`` key as
    plain values; ``on_round`` keys through its :func:`_hook_sig`
    proxy — see the module docstring's *Cache key* section.
    """
    rules = list(rules)
    payload = {
        "version": CACHE_VERSION,
        "ruleset": ruleset_fingerprint(rules),
        "seeds": seed_fingerprint(seed_terms),
        "params": {
            "fuel": fuel,
            "numeric_check": numeric_check,
            "require_overlap": require_overlap,
            "emit_subsumed": emit_subsumed,
            "rounds": rounds,
            "max_emitted": max_emitted,
            "observer": _hook_sig(on_round),
        },
    }
    return _sha([json.dumps(payload, sort_keys=True)])


# ---------------------------------------------------------------------------
#  RuleCache + convenience wrapper
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    env = os.environ.get("MLOPT_RULECACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "mlopt" / "rulecache"


class RuleCache:
    """Filesystem cache of synthesized rule sets, one JSON file per key.

    Parallel safety (plan 0032 phase 2b): keys are content-addressed,
    so two workers computing the same problem write the *same* bytes —
    ``store`` writes to a per-process ``.tmp-<pid>-<uuid>`` name and
    ``os.replace``es it into place, making each commit atomic and the
    last-writer-wins race harmless.  No lockfile is needed; a lock
    would only serialize identical work.  ``store`` also sweeps stale
    ``.tmp-*`` siblings a crashed writer left behind — but only ones
    older than :data:`_TMP_MIN_AGE_S`, so a live writer's in-flight
    tmp file is never swept out from under it.
    """

    #: Tmp files younger than this are in-flight writes, not orphans.
    _TMP_MIN_AGE_S = 60.0

    def __init__(self, cache_dir: Any):
        """Initialise the cache directory."""
        self.dir = Path(cache_dir)

    def path(self, key: str) -> Path:
        """Return the cache file path for ``key``."""
        return self.dir / f"rules-{key}.json"

    def _sweep_tmp(self) -> None:
        """Remove old tmp files a crashed writer left in the cache dir."""
        now = time.time()
        for tmp in self.dir.glob(".rules-*.tmp-*"):
            with contextlib.suppress(OSError):
                if now - tmp.stat().st_mtime > self._TMP_MIN_AGE_S:
                    tmp.unlink()

    def store(self, key: str, rules: Iterable[Rewrite]) -> Path:
        """Persist *rules* under *key*.

        Raises ``TypeError`` if a rule carries hooks without a
        ``guard_pats`` spec (not synthesizable-output), rather than
        silently dropping its side conditions.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        self._sweep_tmp()
        payload = {
            "version": CACHE_VERSION,
            "key": key,
            "rules": [_enc_rule(r) for r in rules],
        }
        # Per-writer tmp name: two parallel workers never share the
        # buffer file, and the rename publishes a complete file.
        tmp = (
            self.dir
            / f".rules-{key}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        tmp.write_text(json.dumps(payload, sort_keys=True))
        os.replace(tmp, self.path(key))
        return self.path(key)

    def load(
        self, key: str, rules: Iterable[Rewrite] = ()
    ) -> list[Rewrite] | None:
        """Load the rule set stored under *key*, or ``None`` on a miss.

        *rules* is the parent ruleset: guarded rules rebuild their
        composite check/derive from the parents' hooks, resolved by
        name.  Records decode in stored order and each decoded rule
        joins the resolver, so a fixpoint mint's ``syn_*`` parents —
        earlier records of the same payload (synthesis emits a rule's
        parents before it) — resolve payload-internally and a
        multi-round output round-trips self-contained.  A caller rule
        and a stored record sharing a name resolve to the stored one:
        it is the exact object the composite was minted against.  A
        missing parent, a version/key mismatch, or a corrupt file is a
        clean miss — never a partially-loaded set.
        """
        p = self.path(key)
        if not p.exists():
            return None
        try:
            payload = json.loads(p.read_text())
            if (
                payload.get("version") != CACHE_VERSION
                or payload.get("key") != key
            ):
                return None
            by_name = {r.name: r for r in rules}
            out = []
            for rec in payload["rules"]:
                rule = _dec_rule(rec, by_name)
                by_name[rule.name] = rule
                out.append(rule)
            return out
        except Exception:
            return None


def synthesize_rules_cached(
    rules: list[Rewrite],
    seed_terms: Iterable[Any] = (),
    *,
    fuel: int = 512,
    numeric_check: bool = True,
    require_overlap: bool = True,
    emit_subsumed: bool = False,
    rounds: int = 1,
    max_emitted: int | None = None,
    on_round: Callable[[int, list[Rewrite]], object] | None = None,
    cache_dir: Any = None,
) -> list[Rewrite]:
    """``meta.synthesize_rules`` with persistence.

    On a repeat call with the same ruleset, seeds, and parameters the
    derived rules are loaded from *cache_dir* (composite guards rebuilt
    from the parent hooks) instead of re-derived.

    ``rounds`` / ``max_emitted`` / ``on_round`` are the fixpoint knobs
    of :func:`mlopt_core.meta.synthesize_rules`; both scalar knobs
    join the cache key and the observer keys through its
    :func:`_hook_sig` proxy, so a ``rounds=3`` entry never collides
    with the ``rounds=1`` mint.  Note the hit semantics: *on_round
    observes a live mint only* — a warm load replays no rounds, so the
    observer is never called on a cache hit (callers measuring strata
    counts should not read the hit as an empty fixpoint).

    Loaded rules are ordinary ``Rewrite`` objects — they fire in
    e-graphs, carry ``.parents`` provenance, and re-register in
    ``meta.SYNTH_PARENTS``.  ``cache_dir`` defaults to
    ``$MLOPT_RULECACHE_DIR`` or ``~/.cache/mlopt/rulecache``.
    """
    rules = list(rules)
    cache = RuleCache(cache_dir or _default_cache_dir())
    key = cache_key(
        rules,
        seed_terms,
        fuel=fuel,
        numeric_check=numeric_check,
        require_overlap=require_overlap,
        emit_subsumed=emit_subsumed,
        rounds=rounds,
        max_emitted=max_emitted,
        on_round=on_round,
    )
    hit = cache.load(key, rules)
    if hit is not None:
        return hit
    derived = M.synthesize_rules(
        rules,
        seed_terms,
        fuel=fuel,
        numeric_check=numeric_check,
        require_overlap=require_overlap,
        emit_subsumed=emit_subsumed,
        rounds=rounds,
        max_emitted=max_emitted,
        on_round=on_round,
    )
    cache.store(key, derived)
    return derived
