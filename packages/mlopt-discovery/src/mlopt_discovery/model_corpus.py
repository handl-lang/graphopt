"""Per-model corpus — a target model mints its own coverage probes.

The discovery corpus is fixed: the bench law cases, the
``mlopt_torch.models`` registry and the intake/zoo side-files.  A
*new* model — one nobody registered — used to be measured only
through zoo analogues: ``play._zoo_case``/``gen_probe`` run the
claim → measured-cost referee → deliver → verify → time pipeline,
but only over ``zoo.zoo()``'s 73 hand-picked workloads.  This module
closes that gap for an arbitrary ``(model, feed)`` pair:

1. **census** — export the model through the same
   ``TorchSource().to_ir`` boundary every corpus term crossed, then
   diff its op-tuples and shapes against the standing corpus
   (:func:`mlopt_discovery.census.op_tuple_census` /
   ``shape_census``).  The report names the ``(op, child-op)``
   tuples the corpus never contained — the patterns only this
   model exercises.
2. **generate** — mutate the model's own term with
   :func:`mlopt_discovery.workload_gen.mutant_term`, the census's
   generative machinery, with the donor statistics mined over
   *corpus + model* so grafts can pull the model's own novel shapes
   back into new compositions.  Every candidate passes
   ``workload_gen.valid_term``'s five gates (mint contract, typing,
   sink-supported ops, torch concrete-eval, novelty vs the corpus
   *and* the model itself) before it becomes a ``TermCase``.
3. **probe** — each case (the model itself first, then its
   mutants) goes through the *same* intake tail ``play._zoo_case``
   runs — ``_mint_tiers`` over the term's real sites, the
   ``_claim_measurable`` split, ``genkernel.profitable``'s
   measured-cost referee on the case's real values, ``gen_sink`` —
   and then ``play.gen_probe``'s claim → deliver → verify → time
   verdict.  Pay claims are measured on the actual model shapes,
   not zoo analogues.

Honesty boundaries (what this does *not* do):

* **The ``optimize`` seam is caller-injected.**  ``mlopt_orchestrator``
  (the eqsat substrate repo) never imports ``mlopt_discovery`` — the
  hexagonal boundary — so ``Optimizer.optimize``/``search`` take an
  optional ``corpus_probe=`` callable invoked once after export as
  ``probe(name, model, x)``; its report lands in
  ``stats["corpus_probe"]`` and the default (``None``) path is
  unchanged.  :func:`corpus_probe` below is the drop-in adapter —
  ``opt.optimize(model, x, corpus_probe=mc.corpus_probe(
  n_mutants=8, reps=50))``.  A plain ``model_corpus`` also satisfies
  the ``(name, model, feed)`` signature directly.
* **Static shapes only.**  The export boundary needs concrete
  example inputs; dynamic shapes (``None`` dims, data-dependent
  structure, ``torch.compile``-only constructs) reject at
  ``torch.export`` and are recorded as rejections, exactly like
  ``intake``'s ledger — never silently skipped.
* **Generated workloads are probes, not census members.**  The
  mutants inform this run's verdict table; they are not appended
  to ``intake_corpus.json`` (that side-file is the *ingested
  models* ledger — a generated term has no ``nn.Module`` provenance
  to record honestly).  ``--write-intake`` is deliberately absent.

Run::

    .venv/bin/python -m mlopt_discovery.model_corpus \
        --module my_pkg.models:build --mutants 8 --reps 50
    .venv/bin/python -m mlopt_discovery.model_corpus \
        --zoo GatedRecurrentPool --mutants 4 --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from mlopt_core.ir import IR
from mlopt_torch.adapters import TorchSink, TorchSource

from mlopt_discovery.impact import TermCase, _ir_of

__all__ = [
    "corpus_probe",
    "coverage_report",
    "export_ir",
    "generate_mutants",
    "main",
    "model_board",
    "model_corpus",
    "mutant_board",
]

_DEV = torch.device("cpu")


# ---------------------------------------------------------------------------
#  Export — the same boundary every corpus term crossed
# ---------------------------------------------------------------------------


def _as_dtype(obj: Any, td: Any) -> Any:
    """Cast tensors in *obj* to *td*; pass non-tensors through."""
    if torch.is_tensor(obj):
        return obj.to(td)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_as_dtype(x, td) for x in obj)
    return obj


def export_ir(
    model: Any, feed: Any, *, dtype: str = "fp64"
) -> tuple[IR, dict, tuple]:
    """Export ``model`` on ``feed``; return ``(ir, leaves, feed)``.

    ``leaves`` is the source-tensor map ``TorchSource.to_ir`` returns
    (param/buffer values keyed by IR name) — the same dict
    ``play._zoo_case`` seats as ``ex["leaves"]`` and feeds to
    ``_var_env``/``profitable``.  ``feed`` is normalised to a tuple
    positional to ``ir.inputs``.  ``dtype`` selects the intake
    convention (``"fp64"`` default, ``"fp32"`` for shipped models) —
    the module is moved in place, exactly as ``_zoo_case`` does.
    """
    td = {"fp32": torch.float32, "fp64": torch.float64}[dtype]
    model = model.to(td)
    feed = _as_dtype(feed, td)
    ir, leaves = TorchSource().to_ir(model, feed)
    feeds = feed if isinstance(feed, tuple) else (feed,)
    return ir, dict(leaves), tuple(feeds)


def _term_case(
    name: str, ir: IR, leaves: dict, feed: tuple
) -> TermCase:
    """Wrap an exported ``(ir, leaves, feed)`` as a ``TermCase``."""
    return TermCase(
        source="model",
        name=name,
        term=ir.root,
        inputs=tuple(ir.inputs),
        feed=feed,
        param_vals=dict(leaves),
    )


# ---------------------------------------------------------------------------
#  Census — which op-tuples/shapes only this model contains
# ---------------------------------------------------------------------------


def _as_corpus(terms: Any) -> list:
    """Normalise corpus entries to ``CorpusTerm``-shaped objects."""
    from mlopt_discovery.census import CorpusTerm

    out = []
    for c in terms:
        if isinstance(c, CorpusTerm):
            out.append(c)
        else:
            out.append(CorpusTerm(c.source, c.name, c.term))
    return out


def default_corpus() -> list:
    """Return the standing corpus: bench + models + intake."""
    from mlopt_discovery import census as _census

    return _census.corpus()


def coverage_report(model_tc: TermCase, corpus: list) -> dict:
    """Diff *model_tc*'s census keys against *corpus*.

    Returns the model's own op-tuple/shape totals plus the tuples and
    shapes the corpus never contained — the "this model is novel
    here" measurement, on the census's own keys so it composes with
    ``workload_gen.census_delta``.
    """
    from mlopt_discovery.census import (
        CorpusTerm,
        op_tuple_census,
        shape_census,
        shape_repr,
    )

    base_op, _ = op_tuple_census(_as_corpus(corpus))
    base_sh, _ = shape_census(_as_corpus(corpus))
    mine = [CorpusTerm("model", model_tc.name, model_tc.term)]
    my_op, _ = op_tuple_census(mine)
    my_sh, _ = shape_census(mine)
    novel_tuples = sorted(k for k in my_op if k not in base_op)
    novel_shapes = sorted(k for k in my_sh if k not in base_sh)
    return {
        "n_model_terms": 1,
        "model_op_tuples": len(my_op),
        "model_shapes": len(my_sh),
        "corpus_terms": len(corpus),
        "novel_op_tuples": [
            {"op": k[0], "children": list(k[1]), "count": my_op[k]}
            for k in novel_tuples
        ],
        "n_novel_op_tuples": len(novel_tuples),
        "novel_shapes": [shape_repr(k) for k in novel_shapes],
        "n_novel_shapes": len(novel_shapes),
    }


# ---------------------------------------------------------------------------
#  Generation — mutants of the model's own term
# ---------------------------------------------------------------------------


def generate_mutants(
    model_tc: TermCase,
    corpus: list,
    n: int,
    *,
    seed: int = 0,
    supported: frozenset | None = None,
    max_nodes: int | None = None,
) -> tuple[list[TermCase], dict]:
    """Mutate *model_tc*'s term into *n* verified probe workloads.

    Donor statistics are mined over ``corpus + model`` — the model's
    own novel subterms join the graft/lift pools, so a mutant can
    recombine the model's patterns into shapes the corpus never
    saw.  Every candidate passes ``workload_gen.valid_term``'s five
    gates; the returned stats record attempts/accepts/rejects for
    the honesty table.

    ``max_nodes`` overrides ``workload_gen._MAX_NODES`` for the
    generation loop only: the corpus-wide 80-node budget assumes
    corpus-sized terms, but a real model's mutants legitimately run
    larger — the caller passes ``model_nodes + headroom``.  The
    override is restored in ``finally``; it is a probe-size bound,
    not a relaxation of the typing/eval/novelty gates.
    """
    from mlopt_discovery import workload_gen as wg
    from mlopt_discovery.census import shape_key

    supported = (
        TorchSink().supported_ops if supported is None else supported
    )
    st = wg.corpus_stats([*_as_corpus(corpus), model_tc])
    rng = random.Random(seed)
    seen: set = set()
    out: list[TermCase] = []
    stats = {"attempted": 0, "accepted": 0, "rejected": {}}
    max_attempts = max(n * 30, 60)
    prev_cap = wg._MAX_NODES
    if max_nodes is not None:
        wg._MAX_NODES = max_nodes
    try:
        for _ in range(max_attempts):
            if len(out) >= n:
                break
            stats["attempted"] += 1
            cand = wg.mutant_term(model_tc, st, rng)
            if cand is None:
                stats["rejected"]["mint"] = (
                    stats["rejected"].get("mint", 0) + 1
                )
                continue
            env = wg.valid_term(cand, st, seen, supported)
            if env is None:
                stats["rejected"]["gate"] = (
                    stats["rejected"].get("gate", 0) + 1
                )
                continue
            c = wg.term_to_case(
                cand, f"gen-model:{len(out)}", "gen-model", env
            )
            if c is None:
                stats["rejected"]["case"] = (
                    stats["rejected"].get("case", 0) + 1
                )
                continue
            seen.add(shape_key(cand))
            out.append(c)
            stats["accepted"] += 1
    finally:
        wg._MAX_NODES = prev_cap
    return out, stats


# ---------------------------------------------------------------------------
#  Board intake — play._case_tail, the shared ``_zoo_case`` tail
# ---------------------------------------------------------------------------
#
#  ``play._case_tail`` (extracted from ``_zoo_case``) is the intake
#  tail on an already-lifted term: ``_mint_tiers`` over the term's
#  real sites → ``_claim_measurable`` split →
#  ``genkernel.profitable`` measured-cost referee → ``gen_sink``.
#  ``_zoo_case`` and this module's boards both call it, so the mint,
#  referee and sink cannot drift.


def _case_tail(
    ir: IR,
    leaves: dict,
    feed: tuple,
    model: Any,
    label: str,
    *,
    gen_reps: int = 20,
) -> tuple:
    """Run ``play._case_tail`` on an already-lifted term.

    Returns the gen-domain case tuple ``(name, term, decls, tag,
    supported, ex)`` — indices ``gen_probe`` reads are identical:
    ``case[1]`` the IR root, ``case[4]`` the supported bound,
    ``case[5]`` the ``ex`` record carrying ``model``/``input``/
    ``ir``/``leaves``/``gen_handlers``/``sink``.  *label* fills both
    label slots (the zoo passes distinct ``case``/``tag`` names).
    """
    from mlopt_discovery import play

    return play._case_tail(
        ir, leaves, feed, model, label, label, gen_reps=gen_reps
    )


def model_board(
    name: str,
    model: Any,
    feed: Any,
    *,
    dtype: str = "fp64",
    gen_reps: int = 20,
) -> tuple[tuple, TermCase, dict]:
    """Export *model* and build its gen-domain board — row 0.

    The model's *own* eager module is seated as ``ex["model"]`` (the
    baseline ``gen_probe`` times), exactly as ``_zoo_case`` seats the
    workload's module — pay claims are priced against the real
    forward, not a re-lowered spelling.
    """
    ir, leaves, feeds = export_ir(model, feed, dtype=dtype)
    tc = _term_case(name, ir, leaves, feeds)
    case = _case_tail(
        ir, leaves, feeds, model, f"model:{name}", gen_reps=gen_reps
    )
    return case, tc, {"ir": ir, "leaves": leaves, "feed": feeds}


def mutant_board(
    tc: TermCase, *, label: str, gen_reps: int = 20
) -> tuple | None:
    """Build a gen-domain board for a generated ``TermCase``.

    There is no ``nn.Module`` to seat — the baseline ``gen_probe``
    times is the term's own generic lowering (``TorchSink.lower`` →
    ``IRModule``), which *is* the spelled denotation the delivered
    kernel is verified against.  Returns ``None`` when the term does
    not lower (an honest decline row is the caller's job — the
    generator's gates make this rare).
    """
    ir = _ir_of(tc.term, tc.inputs)
    leaves = dict(tc.param_vals)
    try:
        model = TorchSink().lower(ir, tc.param_vals)
    except Exception:
        return None
    return _case_tail(
        ir, leaves, tc.feed, model, label, gen_reps=gen_reps
    )


# ---------------------------------------------------------------------------
#  The driver — census, generate, probe, verdict
# ---------------------------------------------------------------------------


def _verdict(row: dict) -> str:
    """Classify a ``gen_probe`` row: pay / decline / error."""
    if not row.get("delivered"):
        return "decline"
    if not row.get("verified"):
        return "verify-fail"
    sp = row.get("speedup")
    if sp is not None and sp > 1.0:
        return "pay"
    return "decline"


def model_corpus(
    name: str,
    model: Any,
    feed: Any,
    *,
    n_mutants: int = 8,
    seed: int = 0,
    corpus: list | None = None,
    dtype: str = "fp64",
    reps: int = 50,
    gen_reps: int = 20,
    probe: bool = True,
    mutant_headroom: int = 60,
) -> dict:
    """Run the per-model corpus pipeline; return the report.

    ``corpus`` defaults to :func:`default_corpus` (bench + models +
    intake side-file) — inject a small list in tests.  ``probe=False``
    stops after census + generation + board intake (no ``gen_probe``
    timing), for callers that only want the coverage measurement.
    ``mutant_headroom`` is the node budget mutants may grow past the
    model's own size (``model_nodes + headroom``, floored at
    ``workload_gen._MAX_NODES``) — see :func:`generate_mutants`.
    """
    corpus = default_corpus() if corpus is None else list(corpus)
    case0, tc, _exp = model_board(
        name, model, feed, dtype=dtype, gen_reps=gen_reps
    )
    cov = coverage_report(tc, corpus)
    from mlopt_discovery import workload_gen as _wg
    from mlopt_discovery.impact import _iter_subterms

    node_cap = max(
        _wg._MAX_NODES,
        min(240, len(_iter_subterms(tc.term)) + mutant_headroom),
    )
    mutants, mstats = (
        generate_mutants(
            tc, corpus, n_mutants, seed=seed, max_nodes=node_cap
        )
        if n_mutants > 0
        else ([], {"attempted": 0, "accepted": 0, "rejected": {}})
    )

    boards: list[tuple] = [case0]
    board_notes: list[dict] = []
    for i, mtc in enumerate(mutants):
        b = mutant_board(
            mtc, label=f"gen:{name}:{i}", gen_reps=gen_reps
        )
        if b is None:
            board_notes.append(
                {
                    "case": f"gen:{name}:{i}",
                    "delivered": False,
                    "note": "lower declined",
                }
            )
            continue
        boards.append(b)

    rows: list[dict] = []
    if probe:
        from mlopt_discovery import play

        for b in boards:
            try:
                rows.append(play.gen_probe(b, reps=reps))
            except Exception as exc:  # honest per-case failure row
                rows.append(
                    {
                        "case": b[0],
                        "delivered": False,
                        "note": (
                            f"probe raised: {type(exc).__name__}: {exc}"
                        ),
                    }
                )
    for n in board_notes:
        rows.append(n)

    verdicts = {v: 0 for v in ("pay", "decline", "verify-fail")}
    for r in rows:
        verdicts[_verdict(r)] = verdicts.get(_verdict(r), 0) + 1
    return {
        "name": name,
        "coverage": cov,
        "mutants": {
            "requested": n_mutants,
            "accepted": len(mutants),
            **mstats,
        },
        "n_boards": len(boards),
        "rows": rows,
        "verdicts": verdicts,
    }


def _model_dtype(model: Any) -> str:
    """Match the intake dtype to *model*'s floating-point parameters.

    ``"fp32"`` for an fp32 module, ``"fp64"`` otherwise — the probe's
    ``export_ir`` casts the module in place, so matching the caller's
    dtype keeps the ``corpus_probe=`` hook from silently re-dtyping
    the model ``lower``'s verify then runs against the original
    ``x``.  Non-tensor/parameterless models fall back to ``"fp64"``
    (the intake convention).
    """
    try:
        for p in model.parameters():
            if torch.is_tensor(p) and p.is_floating_point():
                return "fp32" if p.dtype == torch.float32 else "fp64"
    except Exception:
        pass
    return "fp64"


def corpus_probe(
    *,
    dtype: str | None = None,
    **kw: Any,
) -> Callable[[str, Any, Any], dict]:
    """Return a ``corpus_probe=`` callable for ``Optimizer.optimize``.

    The eqsat orchestrator stays backend-neutral — it cannot import
    this module — so the seam is an injected ``(name, model, x) ->
    report`` callable invoked once after export; the report lands in
    ``stats["corpus_probe"]``.  This factory binds :func:`model_corpus`'s
    knobs (``n_mutants``, ``seed``, ``corpus``, ``reps``, ``gen_reps``,
    ``probe``, ``mutant_headroom``) into that signature::

        opt = Optimizer(backend=TorchBackend())
        mod, stats = opt.optimize(
            model, x,
            corpus_probe=mc.corpus_probe(n_mutants=8, reps=50),
        )
        rep = stats["corpus_probe"]   # the model_corpus report

    ``dtype`` defaults to :func:`_model_dtype` — the model's own
    floating-point dtype — so the intake cast is a no-op on the
    caller's module; pass ``"fp64"`` explicitly for the zoo-table
    convention (which DOES re-dtype an fp32 model — the caller owns
    that side effect, exactly as a direct ``model_corpus`` call does).
    ``model_corpus`` itself is also a valid ``corpus_probe=`` value
    (its ``(name, model, feed)`` signature matches) when the defaults
    suit.
    """

    def call(name: str, model: Any, feed: Any) -> dict:
        options = dict(kw)
        options["dtype"] = (
            dtype if dtype is not None else _model_dtype(model)
        )
        return model_corpus(name, model, feed, **options)

    return call


# ---------------------------------------------------------------------------
#  CLI — an arbitrary model by registry name or module callable
# ---------------------------------------------------------------------------


def _resolve_workload(spec: str) -> tuple[str, Any, Any]:
    """Resolve ``zoo:NAME`` / ``intake:NAME`` / bare zoo name."""
    if ":" in spec:
        kind, name = spec.split(":", 1)
    else:
        kind, name = "zoo", spec
    if kind == "zoo":
        from mlopt_discovery import zoo as _zoo

        for wl in _zoo.zoo():
            if wl.name == name:
                model, feed = wl.build()
                return name, model, feed
        raise ValueError(f"unknown zoo workload: {name!r}")
    if kind == "intake":
        from mlopt_discovery import intake as _intake

        for wl in _intake.candidates():
            if wl.name == name:
                model, feed = wl.build()
                return name, model, feed
        raise ValueError(f"unknown intake workload: {name!r}")
    raise ValueError(f"unknown workload kind: {kind!r}")


def _resolve_module(spec: str) -> tuple[str, Any, Any]:
    """Import ``pkg.mod:callable`` → ``(name, model, feed)``.

    The callable takes no arguments and returns ``(model, feed)`` —
    or a bare ``nn.Module``, in which case ``--input`` must supply a
    shape spec ``"2,8,16"`` (or ``"2,8,16;2,8,16"`` for a tuple of
    feeds, ``;``-separated).
    """
    mod_name, _, fn = spec.partition(":")
    if not fn:
        raise ValueError("--module needs 'pkg.mod:callable'")
    import importlib

    fn_obj = getattr(importlib.import_module(mod_name), fn)
    out = fn_obj()
    if isinstance(out, tuple) and len(out) == 2:
        model, feed = out
    else:
        model, feed = out, None
    return fn, model, feed


def _input_from_spec(spec: str) -> Any:
    """Parse ``"2,8,16"`` / ``"2,8,16;4,8"`` into tensors/tuple."""
    shapes = [
        tuple(int(d) for d in part.split(",") if part.strip())
        for part in spec.split(";")
    ]
    tensors = [torch.randn(s, dtype=torch.float64) for s in shapes]
    return tensors[0] if len(tensors) == 1 else tuple(tensors)


def main(argv: list[str] | None = None) -> int:
    """Census + generate + probe one model; print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--workload",
        help="'zoo:NAME', 'intake:NAME', or a bare zoo name",
    )
    src.add_argument(
        "--module",
        help="'pkg.mod:callable' returning (model, feed) or a module",
    )
    parser.add_argument(
        "--input",
        help="feed shape spec for --module bare modules: '2,8,16'",
    )
    parser.add_argument("--name", help="report name override")
    parser.add_argument("--mutants", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--gen-reps", type=int, default=20)
    parser.add_argument(
        "--dtype", choices=("fp32", "fp64"), default="fp64"
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="census + generation + intake only (no gen_probe timing)",
    )
    parser.add_argument(
        "--corpus",
        choices=("full", "bench"),
        default="full",
        help="donor corpus: full (bench+models+intake) or bench only",
    )
    parser.add_argument("--json", help="write machine-readable report")
    args = parser.parse_args(argv)

    if args.workload:
        name, model, feed = _resolve_workload(args.workload)
    else:
        name, model, feed = _resolve_module(args.module)
        if feed is None:
            if not args.input:
                raise SystemExit(
                    "--module returned a bare module; pass --input "
                    "'d0,d1,...' (or 'd0,d1;d2,d3' for multi-input)"
                )
            feed = _input_from_spec(args.input)
    name = args.name or name

    corpus = None
    if args.corpus == "bench":
        from mlopt_discovery.impact import _bench_cases

        corpus = _bench_cases()[0]

    rep = model_corpus(
        name,
        model,
        feed,
        n_mutants=args.mutants,
        seed=args.seed,
        corpus=corpus,
        dtype=args.dtype,
        reps=args.reps,
        gen_reps=args.gen_reps,
        probe=not args.no_probe,
    )

    cov = rep["coverage"]
    print(  # stdout-compat
        f"== model_corpus — {name} ==\n"
        f"   corpus: {cov['corpus_terms']} terms; model carries "
        f"{cov['model_op_tuples']} op-tuples / {cov['model_shapes']} "
        f"shapes\n"
        f"   novel vs corpus: {cov['n_novel_op_tuples']} op-tuples, "
        f"{cov['n_novel_shapes']} shapes"
    )
    for t in cov["novel_op_tuples"][:20]:
        print(  # stdout-compat
            f"     {t['count']}x  {t['op']}({', '.join(t['children'])})"
        )
    m = rep["mutants"]
    print(  # stdout-compat
        f"   mutants: {m['accepted']}/{m['requested']} accepted "
        f"({m['attempted']} attempts; rejects {m['rejected']})"
    )
    print(f"   verdicts: {rep['verdicts']}")  # stdout-compat
    for r in rep["rows"]:
        print(  # stdout-compat
            f"     {r.get('case')}: delivered={r.get('delivered')} "
            f"verified={r.get('verified')} "
            f"speedup={r.get('speedup')} "
            f"vs_compiled={r.get('vs_compiled')} "
            f"{r.get('note', '')}"
        )
    if args.json:
        Path(args.json).write_text(
            json.dumps(rep, indent=2, default=str) + "\n"
        )
        print(f"wrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
