"""Audit shipped ``dspec`` derives and measure the procedural residual.

Two halves of the derive-as-data audit (ADR 0004), measured over the
**deepened domain** — ``object_synthesis._measure_domain(deep=True)``,
which unions the carrier op-term window into the binding envs so a
bound metavariable can resolve to a compound carrier member (the only
condition under which a ``("shape", T)`` read diverges from the
carrier-aware ``("vshape", T)`` read).

* ``--audit`` — every shipped ``dspec`` is compared to its procedural
  oracle on every measured env (outputs *and* vetoes).  The oracles
  are the pre-migration hooks, recovered from git history
  (:func:`recover_oracle`) — ``matmul_t_concat``'s twin survives
  in-tree as ``om._derive_score_concat_dim``.
* ``--residual`` — every remaining procedural ``derive`` is measured
  (sampled envs, per-attr sample counts, veto share) and fitted by the
  *bounded* synthesizer (:func:`bounded_synth`): the shipped
  candidate space (same-level products, deduped prefixes) with the
  deepest level streamed early-bailed against the expected vector —
  the blind product is quartic in the level size (measured: a 1060-
  member level-1 makes ~7.9 M pairs; materializing their 100+-long
  signatures is what hung the unbounded run), so the last level is
  evaluated lazily and cut by a wall-clock budget.  A truncation is
  reported as ``budget``, never silently.

Usage::

    .venv/bin/python lab/audit/dspec_audit.py --audit
    .venv/bin/python lab/audit/dspec_audit.py --residual [--leaf]

Both need a ``git`` binary on PATH (oracle recovery runs
``git show <rev>:<path>`` inside this repo).
"""

from __future__ import annotations

import argparse
import ast
import signal
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

#: Commits whose parents carry the pre-migration procedural hooks.
REV_CORE = "34eb570^"  # laws: dspec — declarative derive=, core laws
REV_CARRIERS = "4f586f8^"  # derive-as-data: carrier hooks migrate

TENSOR_PY = "packages/catopt-core/src/catopt_core/laws/tensor.py"
SCAN_PY = "packages/catopt-core/src/catopt_core/laws/scan.py"
OM_PY = "packages/catopt-carriers/src/catopt_carriers/om.py"
TRACE_PY = "packages/catopt-carriers/src/catopt_carriers/trace.py"
DECODE_LAWS_PY = (
    "packages/catopt-carriers/src/catopt_carriers/decode_laws.py"
)
DECODE_GEOM_PY = (
    "packages/catopt-carriers/src/catopt_carriers/decode_geom.py"
)

#: (revision, file, function) for each dspec rule's procedural oracle.
#: ``lambda:``-entries were anonymous hooks at the call site.
ORACLE_TABLE: dict[str, tuple[str, str, str]] = {
    # core laws/tensor.py @34eb570^
    "softmax_fold": (REV_CORE, TENSOR_PY, "_derive_softmax_dim"),
    "rms_norm_fold": (REV_CORE, TENSOR_PY, "_derive_rms_norm"),
    "rms_norm_fold_nogain": (REV_CORE, TENSOR_PY, "_derive_rms_norm"),
    "qkv_fuse_asym": (REV_CORE, TENSOR_PY, "_derive_split_sizes"),
    "sdpa_fold_addmul": (REV_CORE, TENSOR_PY, "_derive_scale_mul"),
    "sdpa_fold_addmul_drop": (REV_CORE, TENSOR_PY, "_derive_scale_mul"),
    "sdpa_fold_masked_fillmul": (
        REV_CORE,
        TENSOR_PY,
        "_derive_scale_mul",
    ),
    "sdpa_fold_masked_fillmul_drop": (
        REV_CORE,
        TENSOR_PY,
        "_derive_scale_mul",
    ),
    "sdpa_fold_adddiv": (REV_CORE, TENSOR_PY, "_derive_scale_div"),
    "sdpa_fold_adddiv_drop": (REV_CORE, TENSOR_PY, "_derive_scale_div"),
    "sdpa_fold_masked_filldiv": (
        REV_CORE,
        TENSOR_PY,
        "_derive_scale_div",
    ),
    "sdpa_fold_masked_filldiv_drop": (
        REV_CORE,
        TENSOR_PY,
        "_derive_scale_div",
    ),
    "sdpa_fold_add": (REV_CORE, TENSOR_PY, "_derive_scale_one"),
    "sdpa_fold_add_drop": (REV_CORE, TENSOR_PY, "_derive_scale_one"),
    "sdpa_fold_masked_fill": (REV_CORE, TENSOR_PY, "_derive_scale_one"),
    "sdpa_fold_masked_fill_drop": (
        REV_CORE,
        TENSOR_PY,
        "_derive_scale_one",
    ),
    "sdpa_fold_nomask": (REV_CORE, TENSOR_PY, "_derive_scale_one"),
    "sdpa_fold_div_nomask": (REV_CORE, TENSOR_PY, "_derive_scale_div"),
    # core laws/scan.py @34eb570^
    "affd_lift_unit": (REV_CORE, SCAN_PY, "_derive_affd_unit"),
    "affd_lift_unit_post": (REV_CORE, SCAN_PY, "_derive_affd_unit"),
    "affd_lift_unit_step": (REV_CORE, SCAN_PY, "_derive_affd_unit"),
    "affd_lift_unit_step_post": (
        REV_CORE,
        SCAN_PY,
        "_derive_affd_unit",
    ),
    # carriers om.py @4f586f8^
    "split_transpose": (REV_CARRIERS, OM_PY, "_derive_split_transpose"),
    "decay_block_diag": (
        REV_CARRIERS,
        OM_PY,
        "_derive_decay_block_diag",
    ),
    "decay_block_cross": (
        REV_CARRIERS,
        OM_PY,
        "_derive_decay_block_cross",
    ),
    # carriers trace.py @4f586f8^
    "tr_vanish_split": (REV_CARRIERS, TRACE_PY, "_derive_vanish_split"),
    "tr_slide": (REV_CARRIERS, TRACE_PY, "_derive_slide"),
    "tr_slide_rev": (REV_CARRIERS, TRACE_PY, "_derive_slide_rev"),
    "tr_expand": (REV_CARRIERS, TRACE_PY, "_derive_expand"),
    # decode_laws.py / decode_geom.py @4f586f8^
    "index_select_dedup": (
        REV_CARRIERS,
        DECODE_LAWS_PY,
        "_derive_dedup_index",
    ),
    "cat_narrow_merge": (
        REV_CARRIERS,
        DECODE_GEOM_PY,
        "_derive_narrow_merge",
    ),
}

#: Module the oracle's free names resolve against (the old hook's own
#: module — its helpers survive post-migration).
ORACLE_MODULE = {
    TENSOR_PY: "catopt_core.laws.tensor",
    SCAN_PY: "catopt_core.laws.scan",
    OM_PY: "catopt_carriers.om",
    TRACE_PY: "catopt_carriers.trace",
    DECODE_LAWS_PY: "catopt_carriers.decode_laws",
    DECODE_GEOM_PY: "catopt_carriers.decode_geom",
}


def _vanish_merge_oracle(bound: dict) -> dict:
    """Mint ``UV := (DU, DV)`` — the anonymous tr_*_merge hook @4f586f8^."""
    return {"$attr:UV": (bound["$attr:DU"], bound["$attr:DV"])}


LAMBDA_ORACLES = {
    "tr_vanish_merge": _vanish_merge_oracle,
    "tr_superpose_rev": _vanish_merge_oracle,
}

#: In-tree oracle — kept deliberately as the migration's audit twin.
INTREE_ORACLES = {
    "matmul_t_concat": (
        "catopt_carriers.om",
        "_derive_score_concat_dim",
    ),
}


def _git_show(rev: str, path: str) -> str:
    """Return ``path``'s content at ``rev`` inside this repo."""
    out = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{rev}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def recover_oracle(rev: str, path: str, func: str) -> Any:
    """Re-materialize a pre-migration derive hook from git history.

    The old ``def`` is extracted by AST and executed in a namespace
    seeded with its own module's *current* names — the helpers the
    hook calls (``_vshape``, ``_diag_attrs``, ``_dim2``, …) are
    deliberately unchanged in-tree, so the recovered hook is the
    pre-migration code over today's helpers.
    """
    import importlib

    module = importlib.import_module(ORACLE_MODULE[path])
    ns = dict(vars(module))
    # Old module-level imports the hooks may name.
    from catopt_core.laws import base as laws_base
    from catopt_core.typing import _broadcast

    ns.setdefault("_shape_of", laws_base._shape_of)
    ns.setdefault("_broadcast", _broadcast)
    tree = ast.parse(_git_show(rev, path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func:
            code = compile(
                ast.Module(body=[node], type_ignores=[]),
                f"<{rev}:{path}>",
                "exec",
            )
            exec(code, ns)
            return ns[func]
    raise LookupError(f"{func} not found in {rev}:{path}")


def collect_rules() -> dict[str, Any]:
    """Collect every shipped ``Rewrite`` (module constants + RuleSets)."""
    import importlib
    import pkgutil

    import catopt_carriers
    import catopt_core
    import catopt_core.laws
    from catopt_core.egraph import Rewrite

    seen: dict[str, Any] = {}
    mods = []
    for m in pkgutil.iter_modules(catopt_core.laws.__path__):
        mods.append(
            importlib.import_module(f"catopt_core.laws.{m.name}")
        )
    for m in pkgutil.iter_modules(catopt_carriers.__path__):
        try:
            mods.append(
                importlib.import_module(f"catopt_carriers.{m.name}")
            )
        except Exception:
            continue
    for m in pkgutil.iter_modules(catopt_core.__path__):
        if m.name == "laws":
            continue
        try:
            mods.append(
                importlib.import_module(f"catopt_core.{m.name}")
            )
        except Exception:
            continue
    for mod in mods:
        for key in dir(mod):
            value = getattr(mod, key)
            if isinstance(value, Rewrite):
                seen.setdefault(value.name, value)
            elif hasattr(value, "rules") and not isinstance(
                value, type
            ):
                try:
                    for r in value.rules:
                        if isinstance(r, Rewrite):
                            seen.setdefault(r.name, r)
                except TypeError:
                    continue
    return seen


class _Timeout(Exception):
    """Per-rule wall-clock bound exceeded."""


def _alarm(seconds: float) -> None:
    def handler(_sig: int, _frame: Any) -> None:
        raise _Timeout()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(max(1, int(seconds)))


def _clear_alarm() -> None:
    signal.alarm(0)


def _call_derive(derive: Any, env: dict) -> Any:
    """Run a derive; an exception is a veto (the firing-path contract)."""
    try:
        return derive(env)
    except Exception:
        return None


def measure_domain(rule: Any, limit: int, deep: bool) -> Any:
    """``_measure_domain`` with the rule's own hooks, ``deep`` opt-in."""
    from catopt_core.egraph import Rewrite
    from catopt_discovery import object_synthesis as osyn

    probe = Rewrite(rule.name, rule.lhs, rule.rhs, derive=rule.derive)
    return osyn._measure_domain(probe, (), limit, deep=deep)


def measure_domain_oracle(
    rule: Any, oracle: Any, limit: int, corpus: Any = ()
) -> Any:
    """Deep domain of *rule*, probed with the *oracle* as the derive."""
    from catopt_core.egraph import Rewrite
    from catopt_discovery import object_synthesis as osyn

    probe = Rewrite(rule.name, rule.lhs, rule.rhs, derive=oracle)
    return osyn._measure_domain(probe, corpus, limit, deep=True)


# ---------------------------------------------------------------------------
#  Hand-built corpus — the binding enumerator's honest boundary leaves
#  several shipped rules with no synthesized site at all (``envs=0``);
#  small concrete LHS instances close the gap through ``real_matches``.
# ---------------------------------------------------------------------------


def hand_corpus(name: str) -> list:
    """Return concrete LHS terms for rules the enumerator cannot bind."""
    from catopt_core.ir import Op, TensorType, Var

    def v(s: str, *shape: int) -> Var:
        return Var(s, TensorType(tuple(shape)))

    def tr(t: Any, u: Any) -> Any:
        return Op.make("trace", t, usize=u)

    if name == "tr_vanish_split":
        return [
            tr(v("f", 8, 8), (2, 3)),
            tr(v("g", 4, 4), (1, 1)),
            tr(v("h", 12, 12), (4, 2)),
        ]
    if name == "tr_vanish_merge":
        return [
            tr(tr(v("f", 8, 8), 2), 3),
            tr(tr(v("g", 4, 4), 1), 1),
            tr(tr(v("h", 2, 2), 3), 3),  # veto: too small
        ]
    if name == "tr_superpose_rev":
        return [
            Op.make("bdiag", tr(v("f", 6, 6), 2), tr(v("g", 9, 9), 3)),
            Op.make("bdiag", tr(v("f", 4, 4), 1), tr(v("g", 4, 4), 1)),
        ]
    if name in ("tr_slide", "tr_slide_rev"):
        eye = Op.make("eye", dim=2)

        def slide_lhs(g: Any, h: Any, usize: int, rev: bool) -> Any:
            blk = Op.make("bdiag", h, eye)
            mm = (
                Op.make("matmul", blk, g)
                if rev
                else Op.make("matmul", g, blk)
            )
            return tr(mm, usize)

        rev = name == "tr_slide_rev"
        return [
            slide_lhs(v("g", 4, 4), v("h", 4, 4), 2, rev),
            slide_lhs(v("g", 6, 6), v("h", 4, 4), 2, rev),
            slide_lhs(v("g", 4, 4), v("h", 4, 4), 8, rev),  # decline
        ]
    if name == "tr_expand":
        return [tr(v("f", 6, 6), 2), tr(v("f", 6, 6), 8)]
    if name == "decay_block_diag":

        def dg(g: Any, t: int, i: int) -> Any:
            s = Op.make(
                "split",
                Op.make("decay", g, t=t),
                sizes=(3, 5),
                dim=-2,
                index=i,
            )
            return Op.make("split", s, sizes=(3, 5), dim=-1, index=i)

        return [
            dg(v("g", 4, 4), 8, 0),
            dg(v("g", 4, 4), 8, 1),
            dg(v("g", 4, 4), 8, 2),
        ]
    if name == "decay_block_cross":

        def dc(g: Any, t: int, i: int, j: int) -> Any:
            s = Op.make(
                "split",
                Op.make("decay", g, t=t),
                sizes=(2, 3, 4),
                dim=-2,
                index=i,
            )
            return Op.make("split", s, sizes=(2, 3, 4), dim=-1, index=j)

        return [
            dc(v("g", 4, 4), 9, 2, 0),
            dc(v("g", 4, 4), 9, 1, 0),
            dc(v("g", 4, 4), 9, 0, 1),
        ]
    if name == "cat_narrow_merge":
        return [
            Op.make(
                "concat",
                Op.make(
                    "narrow", v("t", 4, 10), dim=0, start=0, length=3
                ),
                Op.make(
                    "narrow", v("t", 4, 10), dim=0, start=3, length=4
                ),
                dim=1,
            ),
            Op.make(
                "concat",
                Op.make(
                    "narrow", v("t", 4, 10), dim=1, start=0, length=2
                ),
                Op.make(
                    "narrow", v("t", 4, 10), dim=1, start=2, length=5
                ),
                dim=0,
            ),
        ]
    if name == "index_select_dedup":
        return [
            Op.make(
                "index_select", v("t", 8, 4), dim=0, index=(0, 2, 0, 1)
            ),
            Op.make("index_select", v("t", 8, 4), dim=1, index=(3, 3)),
            Op.make("index_select", v("t", 8, 4), dim=0, index=2),
        ]
    if name == "qkv_fuse_asym":

        def head(w: str, shape: tuple) -> Any:
            return Op.make(
                "transpose",
                Op.make(
                    "reshape",
                    Op.make("linear", v("x", 2, 4, 8), v(w, 8, 8)),
                    shape=shape,
                ),
                dim0=1,
                dim1=2,
            )

        s1, s2, s3 = (2, 4, 4, 2), (2, 4, 4, 2), (2, 4, 4, 2)
        return [
            Op.make(
                "sdpa",
                head("Q", s1),
                head("K", s2),
                head("V", s3),
                scale=0.35,
                enable_gqa=False,
            ),
            Op.make(
                "sdpa",
                head("Q", s1),
                head("K", s2),
                Op.make(
                    "transpose",
                    Op.make(
                        "reshape",
                        Op.make(
                            "linear", v("x", 2, 4, 8), v("V", 16, 8)
                        ),
                        shape=(2, 8, 4, 2),
                    ),
                    dim0=1,
                    dim1=2,
                ),
                scale=0.35,
                enable_gqa=True,
            ),
        ]
    return []


# ---------------------------------------------------------------------------
#  Bounded synthesizer — shipped space, streamed last level
# ---------------------------------------------------------------------------


def _stream_hits(
    level: list[tuple], expected: tuple, deadline: float, state: dict
) -> list[tuple]:
    """Yield-level ``(expr, sig)`` reproducing *expected*, early-bailed.

    The last enumeration level is never materialized: each pair's
    signature is folded position-by-position and abandoned at the
    first divergence from the expected vector.  ``min`` over the hits
    is exact — streaming sees every expr, so the smallest
    ``(size, repr)`` hit is the same one the deduped enumeration
    would mint.
    """
    from catopt_discovery import dspec_synth as dsp

    hits: list[tuple] = []
    n = len(expected)

    def scan(expr_fn_pairs: Any) -> bool:
        """Fold *expr_fn_pairs*; ``False`` when the deadline tripped."""
        for expr, sig_fn in expr_fn_pairs:
            if (
                state["checked"] % 4096 == 0
                and time.monotonic() > deadline
            ):
                state["truncated"] = True
                return False
            state["checked"] += 1
            sig = []
            for i in range(n):
                v = sig_fn(i)
                if v != expected[i]:
                    break
                sig.append(v)
            else:
                hits.append((expr, tuple(sig)))
        return True

    def pairs(op: str) -> Any:
        return (
            (
                (op, ea, eb),
                lambda i, sa=sa, sb=sb, o=op: dsp._vbin(
                    o, sa[i], sb[i]
                ),
            )
            for (ea, sa), (eb, sb) in product(level, level)
        )

    for op in dsp.BINOPS:
        if not scan(pairs(op)):
            return hits
    for op in dsp.UNOPS:
        gen = (
            ((op, ea), lambda i, sa=sa, o=op: dsp._vun(o, sa[i]))
            for ea, sa in level
        )
        if not scan(gen):
            return hits
    gen = (
        (("len", ea), lambda i, sa=sa: dsp._vlen(sa[i]))
        for ea, sa in level
    )
    scan(gen)
    return hits


def _prefix_level(
    level: list[tuple], samples: list[tuple]
) -> list[tuple]:
    """One shipped enumeration level over *level* — deduped."""
    from catopt_discovery import dspec_synth as dsp

    nxt: list[tuple] = []
    for op in dsp.BINOPS:
        for (ea, sa), (eb, sb) in product(level, level):
            nxt.append(
                (
                    (op, ea, eb),
                    tuple(
                        dsp._vbin(op, a, b)
                        for a, b in zip(sa, sb, strict=True)
                    ),
                )
            )
    for op in dsp.UNOPS:
        for ea, sa in level:
            nxt.append(((op, ea), tuple(dsp._vun(op, a) for a in sa)))
    for ea, sa in level:
        nxt.append((("len", ea), tuple(dsp._vlen(a) for a in sa)))
    return dsp._dedupe(nxt)


def bounded_synth(
    samples: list[tuple],
    *,
    name: str | None = None,
    max_depth: int = 2,
    budget_s: float = 120.0,
) -> tuple[dict | None, dict]:
    """``synth_dspec`` with a streamed, deadline-bounded deepest level.

    Same candidate space and same acceptance (``_reads_bound``, exact
    reproduction of every sample) as the shipped synthesizer; the
    prefix levels are materialized and fingerprint-deduped exactly as
    ``_candidates`` does, while the last level streams early-bailed —
    the quartic product stays CPU-bound instead of memory-bound.
    ``meta`` reports ``levels`` sizes, ``checked`` pairs and whether
    the *budget_s* deadline truncated the sweep (``truncated``).
    """
    from catopt_discovery import dspec_synth as dsp

    meta: dict[str, Any] = {
        "checked": 0,
        "truncated": False,
        "levels": [],
    }
    if len(samples) < dsp.MIN_SAMPLES:
        meta["why"] = "thin"
        return None, meta
    values = {v for _b, v in samples}
    if len(values) < 2:
        meta["why"] = "constant"
        return None, meta
    expected = tuple(e for _b, e in samples)
    bound = samples[0][0]
    deadline = time.monotonic() + budget_s
    level = dsp._dedupe(
        [
            (e, dsp._signature(e, samples))
            for e in dsp._leaves(bound, values)
        ]
    )
    meta["levels"].append(len(level))
    hits = [(e, s) for e, s in level if s == expected]
    for _depth in range(max_depth - 1):
        level = _prefix_level(level, samples)
        meta["levels"].append(len(level))
        hits.extend((e, s) for e, s in level if s == expected)
    hits.extend(_stream_hits(level, expected, deadline, meta))
    hits = [(e, s) for e, s in hits if dsp._reads_bound(e)]
    if not hits:
        meta["why"] = "no-fit"
        return None, meta
    best = min(hits, key=lambda c: (dsp._size(c[0]), repr(c[0])))
    return {(name or "OUT"): best[0]}, meta


# ---------------------------------------------------------------------------
#  Audit — shipped dspecs vs their procedural oracles
# ---------------------------------------------------------------------------


def audit_shipped(rules: dict, limit: int) -> list[dict]:
    """Compare every shipped ``dspec`` to its oracle on the deep domain."""
    import importlib

    import catopt_carriers.xcarrier  # noqa: F401  registers _xshape
    from catopt_core.laws.cond import eval_derive

    rows = []
    for name, rule in sorted(rules.items()):
        if getattr(rule, "dspec", None) is None:
            continue
        oracle = None
        source = ""
        if name in ORACLE_TABLE:
            rev, path, func = ORACLE_TABLE[name]
            try:
                oracle = recover_oracle(rev, path, func)
                source = f"{rev}:{func}"
            except Exception as exc:
                source = f"oracle-recovery-failed:{exc}"
        elif name in LAMBDA_ORACLES:
            oracle = LAMBDA_ORACLES[name]
            source = "inline-lambda"
        elif name in INTREE_ORACLES:
            mod, func = INTREE_ORACLES[name]
            oracle = getattr(importlib.import_module(mod), func)
            source = f"in-tree:{func}"
        row = {"rule": name, "oracle": source}
        if oracle is None:
            row["verdict"] = "no-oracle"
            rows.append(row)
            continue
        corpus = hand_corpus(name)
        try:
            _alarm(120)
            dom = measure_domain_oracle(rule, oracle, limit, corpus)
        except _Timeout:
            row["verdict"] = "measure-timeout"
            rows.append(row)
            continue
        finally:
            _clear_alarm()
        mism = []
        fired = vetoed = 0
        for env, _outcome in dom:
            want = _call_derive(oracle, env)
            got = _call_derive(
                lambda b, r=rule: eval_derive(r.dspec, b), env
            )
            if want is None:
                vetoed += 1
            else:
                fired += 1
            if got != want:
                mism.append(
                    {"env": _env_desc(env), "want": want, "got": got}
                )
        row.update(
            envs=len(dom),
            fired=fired,
            vetoed=vetoed,
            divergences=mism[:5],
            n_divergent=len(mism),
        )
        if not dom:
            row["verdict"] = "no-domain"
        elif mism:
            row["verdict"] = "unfaithful"
        elif not fired:
            row["verdict"] = (
                "veto-only"  # agree on declines, never fired
            )
        else:
            row["verdict"] = "faithful"
        rows.append(row)
    return rows


def _env_desc(env: dict) -> str:
    """Short env description for divergence witnesses."""
    from catopt_discovery import oracle as lvo

    return repr(
        sorted(
            (k, lvo._bind_desc(v))
            for k, v in env.items()
            if not k.startswith("$attr:")
        )
        + sorted(
            (k, repr(v))
            for k, v in env.items()
            if k.startswith("$attr:")
        )
    )


# ---------------------------------------------------------------------------
#  Residual — measure the procedural leftovers on leaf + deep domains
# ---------------------------------------------------------------------------


def residual_table(
    rules: dict,
    limit: int,
    *,
    leaf: bool,
    budget_s: float,
    measure_s: float,
) -> list[dict]:
    """Per-rule row: samples, synth outcome, veto agreement, gap."""
    import catopt_carriers.xcarrier  # noqa: F401  registers _xshape
    from catopt_core.laws.cond import eval_derive

    rows = []
    for name, rule in sorted(rules.items()):
        if (
            getattr(rule, "dspec", None) is not None
            or rule.derive is None
        ):
            continue
        row: dict[str, Any] = {"rule": name}
        t0 = time.monotonic()
        try:
            _alarm(measure_s)
            dom = measure_domain(rule, limit, deep=not leaf)
        except _Timeout:
            row["verdict"] = "measure-timeout"
            rows.append(row)
            continue
        finally:
            _clear_alarm()
        bounds = [env for env, _o in dom]
        outcomes = [_call_derive(rule.derive, b) for b in bounds]
        by_attr: dict[str, list] = {}
        veto = 0
        for b, got in zip(bounds, outcomes, strict=True):
            if got is None:
                veto += 1
                continue
            for key, val in got.items():
                k = (
                    key[len("$attr:") :]
                    if key.startswith("$attr:")
                    else key
                )
                by_attr.setdefault(k, []).append((b, val))
        row.update(
            envs=len(dom),
            fired=len(bounds) - veto,
            vetoed=veto,
            attrs=sorted(by_attr),
            samples=sum(len(v) for v in by_attr.values()),
            measure_s=round(time.monotonic() - t0, 1),
        )
        if not by_attr:
            row["verdict"] = "no-domain" if not bounds else "all-veto"
            rows.append(row)
            continue
        spec_all: dict = {}
        attr_rows = []
        fitted = True
        for attr, samples in sorted(by_attr.items()):
            spec, meta = bounded_synth(
                samples, name=attr, budget_s=budget_s
            )
            attr_rows.append(
                {
                    "attr": attr,
                    "samples": len(samples),
                    "fit": spec[attr] if spec else None,
                    "why": meta.get("why"),
                    "levels": meta["levels"],
                    "checked": meta["checked"],
                    "truncated": meta["truncated"],
                }
            )
            if spec is None:
                fitted = False
            else:
                spec_all.update(spec)
        row["attr_fits"] = attr_rows
        if not fitted:
            row["verdict"] = (
                "budget"
                if any(a["truncated"] for a in attr_rows)
                else "declined"
            )
            rows.append(row)
            continue
        # Veto agreement: the spec must decline where the hook did,
        # and must reproduce the full output map where it fired.
        over = under = 0
        for b, got in zip(bounds, outcomes, strict=True):
            spec_out = _call_derive(
                lambda e, s=spec_all: eval_derive(s, e), b
            )
            if got is None and spec_out is not None:
                over += 1
            elif got is not None and spec_out != got:
                under += 1
        row.update(spec=spec_all, overfires=over, misfires=under)
        row["verdict"] = (
            "faithful" if not (over or under) else "unfaithful"
        )
        rows.append(row)
    return rows


def _print_residual(rows: list[dict]) -> None:
    """Render the residual table."""
    tally: dict[str, int] = {}
    for r in rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
        line = (
            f"{r['rule']:42s} {r['verdict']:15s} "
            f"envs={r.get('envs', 0):4d} fired={r.get('fired', 0):4d} "
            f"samples={r.get('samples', 0):4d}"
        )
        if r.get("spec"):
            line += f" spec={r['spec']}"
        if r.get("overfires") or r.get("misfires"):
            line += f" over={r.get('overfires', 0)} mis={r.get('misfires', 0)}"
        print(line)
        for a in r.get("attr_fits", []):
            if a["fit"] is None:
                print(
                    f"    {a['attr']:6s} decline:{a['why']}"
                    f" levels={a['levels']} checked={a['checked']}"
                    f"{' TRUNCATED' if a['truncated'] else ''}"
                )
    print("\ntally:", dict(sorted(tally.items())))


def _print_audit(rows: list[dict]) -> None:
    """Render the shipped-dspec audit table."""
    for r in rows:
        line = (
            f"{r['rule']:42s} {r['verdict']:12s} oracle={r['oracle']}"
        )
        if "envs" in r:
            line += (
                f" envs={r['envs']} fired={r['fired']}"
                f" vetoed={r['vetoed']} divergent={r['n_divergent']}"
            )
        print(line)
        for d in r.get("divergences", [])[:3]:
            print(
                "    DIV",
                d["env"],
                "want:",
                d["want"],
                "got:",
                d["got"],
            )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", action="store_true", help="audit shipped dspecs"
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="measure residual derives",
    )
    parser.add_argument(
        "--leaf",
        action="store_true",
        help="leaf domain (no deep window)",
    )
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--measure-timeout", type=float, default=60.0)
    parser.add_argument(
        "--only", action="append", default=None, metavar="SUBSTR"
    )
    args = parser.parse_args(argv)
    if not (args.audit or args.residual):
        parser.error("pass --audit and/or --residual")
    rules = collect_rules()
    if args.only:
        rules = {
            k: v
            for k, v in rules.items()
            if any(s in k for s in args.only)
        }
    if args.audit:
        _print_audit(audit_shipped(rules, args.limit))
    if args.residual:
        rows = residual_table(
            rules,
            args.limit,
            leaf=args.leaf,
            budget_s=args.budget,
            measure_s=args.measure_timeout,
        )
        _print_residual(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
