"""ProbeBounds — the typed budget object for bounded discovery probes.

The arch audit (``docs/reports/arch_audit_gamekit_discovery.md``
§1.2/§3) found the probe bounds scattered as stringly-typed kwargs
and ``ex[...]`` keys, with two DIFFERENT size metrics answering to
near-identical names.  This module is the single typed object that
replaces them: one frozen dataclass threaded through
``model_corpus`` → ``play._case_tail`` → ``_mint_tiers`` /
``MetaArena`` / ``genkernel.profitable``, plus the mutable
:class:`SkipStats` tally the collectors report honest over-cap
declines into.

THE METRICS — every cap names its unit explicitly
--------------------------------------------------

IR terms are hash-consed DAGs; every consumer below *renders* them
as trees, so a shared subterm is spelled once per occurrence and the
render cost is the tree-expansion, not the distinct-node count
(``census.term_expansion`` / ``census.op_expansion`` measure the two
flavours).  The caps therefore come in two families:

* **spelled tree-size** — ``spec_tree_cap``.  Counts EVERY node a
  spec/pattern render spells: op nodes, metavar/const leaves, and
  one per attrs dict.  This is what the site collectors measure
  (``play._fusion_pattern_sz`` / ``_reduction_pattern`` return it;
  the seq collectors count ``_seq_deindex`` visits — the same
  recurrence).  The minted claim patterns ARE these trees, so this
  is the honest bound on what a claim can usefully bind.
* **op-expansion** — ``spec_op_cap``.  Counts ``Op`` nodes only
  (``census.op_expansion`` — non-``Op`` leaves count 0).  This is
  what ``MetaArena``'s spec inventory bounds: ``_spec_of`` renders
  every op occurrence concrete and abstracts leaves to metavar
  names, so the op count is the render's dominant cost and the
  "pattern too large to claim" proxy.

The two are NOT interchangeable — a fusion pattern abstracts
non-elementwise children to single metavar leaves while
``_spec_of`` spells them, so the same subterm has different sizes
under each metric.  ``ProbeBounds`` keeps them as distinct fields;
the historical ``spec_node_cap`` knob sets BOTH (each measured in
its own unit — the audit's dual-metric hazard was one number
silently meaning two things, not the existence of two bounds).

* ``key_cap`` — ``census.term_expansion`` bound (ops + leaves +
  consts) for novelty ``shape_key``s and the ``valid_term``
  validity gate: a candidate that cannot be keyed cannot be
  checked for novelty, so it declines as unmeasurable.
* ``shape_key_cap`` — ``census.term_expansion`` bound per subterm
  for the coverage census's ``shape_key`` (``None`` → the module
  default ``model_corpus._SHAPE_KEY_CAP``).
* ``max_handlers`` / ``time_budget`` / ``reps`` — the
  ``genkernel.profitable`` measured-cost referee bounds: how many
  minted claims get priced (sorted, deterministic prefix), the
  per-claim paired-window seconds allowance, and the reps override.

``None`` everywhere is the historical unbounded zoo behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["ProbeBounds", "SkipStats"]


class SkipStats(dict):
    """Tally of honest over-cap declines, keyed by collector name.

    A ``dict`` subclass so the existing ``stats.get(k, 0)`` /
    ``stats[k] = …`` collector protocol works unchanged; adds the
    named-counter and merge helpers the reports aggregate with.
    Keys look like ``"fusion_sites_over_cap"``,
    ``"seq_sites_over_cap"``, ``"expansion"`` (mutant rejects) —
    every bounded path reports WHAT it declined and WHY, never a
    silent drop.
    """

    def skip(self, kind: str, n: int = 1) -> None:
        """Increment the *kind* counter by *n*."""
        self[kind] = self.get(kind, 0) + n

    def merge(self, other: Any) -> None:
        """Fold another counter mapping (dict or SkipStats) in."""
        for k, v in dict(other or {}).items():
            self[k] = self.get(k, 0) + int(v)

    @property
    def total(self) -> int:
        """Total declines across all kinds."""
        return sum(int(v) for v in self.values())


@dataclass(frozen=True)
class ProbeBounds:
    """The typed probe budget — see the module docstring for metrics.

    All fields default ``None`` (unbounded — the zoo-scale
    behaviour).  Frozen: a resolved bound set is shared read-only
    by every board a probe run builds.
    """

    #: Spelled tree-size cap for minted claim patterns — the site
    #: collectors' ``spec_cap`` (``_fusion_sites`` /
    #: ``_reduction_sites`` / the seq deindex budgets).
    spec_tree_cap: int | None = None
    #: Op-expansion cap for the ``MetaArena`` spec inventory
    #: (``_composite_specs`` / ``_program_specs``).
    spec_op_cap: int | None = None
    #: Tree-expansion cap (``census.term_expansion``) for novelty
    #: keys and the ``valid_term`` gate.
    key_cap: int | None = None
    #: Tree-expansion cap per subterm for census ``shape_key``s.
    shape_key_cap: int | None = None
    #: ``profitable``: max claims priced per sweep (sorted prefix).
    max_handlers: int | None = None
    #: ``profitable``: seconds per claim's paired timing window.
    time_budget: float | None = None
    #: ``profitable``: per-call-site reps override.
    reps: int | None = None

    # -- derived views -------------------------------------------------

    def referee_dict(self) -> dict:
        """Return the legacy ``{"max_handlers", "time_budget", "reps"}`` dict.

        Kept for the ``ex["_referee"]`` record and any consumer that
        still reads the dict shape; new code should carry the
        :class:`ProbeBounds` itself (``ex["bounds"]``).
        """
        return {
            "max_handlers": self.max_handlers,
            "time_budget": self.time_budget,
            "reps": self.reps,
        }

    def profitable_kwargs(self, *, reps_default: int) -> dict:
        """Keyword args for ``genkernel.profitable``.

        ``reps`` resolves to ``self.reps`` when set, else the
        caller's per-site default (``gen_reps`` at intake, ``20``
        in the board gates).  ``max_handlers``/``time_budget`` pass
        through ``None`` — the referee reads ``None`` as unbounded.
        """
        return {
            "reps": self.reps
            if self.reps is not None
            else reps_default,
            "max_handlers": self.max_handlers,
            "time_budget": self.time_budget,
        }

    def as_dict(self) -> dict:
        """JSON/report echo — metric-explicit field names."""
        return {
            "spec_tree_cap": self.spec_tree_cap,
            "spec_op_cap": self.spec_op_cap,
            "key_cap": self.key_cap,
            "shape_key_cap": self.shape_key_cap,
            "max_handlers": self.max_handlers,
            "time_budget": self.time_budget,
            "reps": self.reps,
        }

    # -- compatibility -------------------------------------------------

    @classmethod
    def from_legacy_ex(cls, ex: dict) -> ProbeBounds:
        """Reconstruct bounds from a case ``ex`` record's legacy keys.

        Cases minted before ``ex["bounds"]`` existed carry
        ``ex["_referee"]`` (the ``{"max_handlers", "time_budget",
        "reps"}`` dict) and ``ex["_spec_node_cap"]`` — the single
        value that fed BOTH the tree-size collectors and the
        op-expansion inventory, so the fallback sets both spec caps
        to it (each then measured in its own unit, exactly as the
        legacy path did).
        """
        ref = ex.get("_referee") or {}
        cap = ex.get("_spec_node_cap")
        return cls(
            spec_tree_cap=cap,
            spec_op_cap=cap,
            max_handlers=ref.get("max_handlers"),
            time_budget=ref.get("time_budget"),
            reps=ref.get("reps"),
        )
