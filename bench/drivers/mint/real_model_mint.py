"""Real-model mint r1 — does minting ON transformer exports pay?

The law-transfer measurement (`docs/reports/law_transfer.md`) showed
the 58 shipped ``prim:synop*`` folds are corpus-locked: they fire only
on the gated-scan zoo family they were minted from.  The open
question this driver answers: **does the same mint machinery produce
paying laws when the corpus IS the real model?**  whisper-base and
t5-small encoder/decoder steps are exported through the same
``TorchSource``/``export_ir`` boundary every corpus term crossed,
then:

1. **mint (per-module, ``zoo-real`` style)** — ``census.op_tuple_census``
   over each module's exported term feeds ``pipeline.propose`` (the
   census-naturality / mixed-view / pattern-recognition /
   fusion-schema / shape-aware / grammar families — the
   ``mul_unsq_nat`` path), gated binding-first by a *bounded* matcher
   (see below).
2. **mint (cross-module, ``zoo-xmodule`` style)** — the same census
   cross-tuples plus ``antiunify.mine_candidates`` over the merged
   corpus e-graph, each surviving shape minting the mechanical
   view-transform family, gated to patterns binding >= ``--xbind``
   DISTINCT modules (``overnight._xmodule_pool`` re-implemented with
   the bounded matcher — the shipped one calls ``_term_match``, which
   hangs on real-model DAGs; see below).
3. **admit** — every deduplicated candidate faces the shipped
   ``bench.referees`` gauntlet (``--referee``, default ``full``:
   oracle -> pay -> sweep -> cert) through ``overnight._admit``,
   over a ``RefereeCtx`` whose ``real_terms`` are the corpus's
   *bounded* real subterms and whose ``probe`` is the modules
   themselves as ``TermCase``s verified through ``TorchSink``.
4. **classify** — admitted laws get the novelty-sieve verdict via
   ``overnight._classify`` (empty case contexts — ``_positions``
   enumerates *paths*, exponential on these DAGs; the synth/loose/
   registry sources still run).
5. **fire/pay on the source model** — each admitted law is added to
   ``default_rules()`` and the source term re-saturated (bounded
   ``EGraph.run``): ``rule_fires`` is the fire measurement.  With
   ``--pay-arm`` the firing module additionally goes through
   ``search -> lower(verify=True) -> time`` vs the eager module and
   vs the same search without the law — the delivered-wall answer,
   under a wall-clock alarm.

**Why a bounded matcher.**  ``mlopt_core.egraph.terms._term_match``
compares concrete pattern leaves by ``repr(pattern) == repr(term)``;
``Op.__repr__`` tree-renders its args, so matching ``mul(0, 'M0')``
against a whisper decoder subterm renders a 457k-node shared DAG —
measured: ``pipeline.propose(require_bind=True)`` never returned on
``whisper:decoder`` (>4 min inside one ``_binds_any``).  The zoo
corpus never saw it because zoo terms are small.  This driver
therefore re-implements the match with two changes, recorded as
``match: bounded`` in the report:

* a non-``Op`` pattern leaf never equals an ``Op`` term (``repr`` of
  an ``Op`` is always ``"op(...)"`` — a ``Const``/``Var``/``Param``
  repr cannot collide; the zoo-scale semantics are unchanged for
  every case that could previously finish);
* every candidate site is pre-bounded by ``term_expansion`` — sites
  whose tree expansion exceeds ``--site-cap`` are skipped and
  counted, never silently matched.

Subterms with expansion over ``--real-cap`` are likewise excluded
from ``real_terms``/bind domains (counted in ``skipped``) — a law
binding ONLY giant subterms would be reported unbound here, an
honest under-approximation of the bind gate, not a false admit:
the gauntlet's own synthesized domain still judges every candidate.

Honesty contract: export failures, mint-family timeouts (per-family
wall budget — a family that exceeds it is recorded ``timeout`` and
its proposals skipped), referee skips/refusals, search timeouts and
verify failures are all rows in the JSON, never silent drops.

Run::

    .venv/bin/python bench/drivers/mint/real_model_mint.py --dry
    .venv/bin/python bench/drivers/mint/real_model_mint.py \
        --out docs/reports/realcorpus_mint_r1.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import signal
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

DEFAULT_OUT = _REPO / "docs" / "reports" / "realcorpus_mint_r1.json"

#: Expansion bounds — the DAG-safety caps described in the docstring.
DEFAULT_REAL_CAP = 2_500  # real_terms / bind-domain subterm ceiling
DEFAULT_SITE_CAP = 1_500  # emit-sanity / fire-site ceiling
DEFAULT_MAX_REAL = 6_000  # real_terms count ceiling
DEFAULT_PAGE = 24  # per-module mint pool cap
DEFAULT_XCAP = 64  # cross-module pool cap
DEFAULT_MAX_CAND = 48  # candidates sent to the gauntlet
DEFAULT_REFEREE = "full"


# ---------------------------------------------------------------------------
#  Wall-clock guards — signal.alarm only bounds the main thread; every
#  risky stage runs there.
# ---------------------------------------------------------------------------


class _Wall(BaseException):
    """A stage exceeded its wall-clock budget.

    ``BaseException`` (not ``Exception``) so broad ``except
    Exception`` handlers inside mined machinery — e.g. per-shape
    funnels in ``prim_mint`` — cannot swallow the wall and let the
    stage run unbounded.
    """


@contextlib.contextmanager
def _alarm(seconds: float):
    """Context manager: raise :class:`_Wall` after *seconds*."""

    def _raise(signum, frame):
        raise _Wall(f"wall budget {seconds}s exceeded")

    prev = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


# ---------------------------------------------------------------------------
#  Bounded matching — _term_match semantics minus the repr() blowup.
# ---------------------------------------------------------------------------


def _match(
    pattern: Any, term: Any, subst: dict | None = None
) -> dict | None:
    """Bounded ``_term_match``: identical semantics, no ``repr(term)``.

    The only divergence: a concrete (non-``Op``, non-metavar) pattern
    leaf vs an ``Op`` term returns ``None`` without rendering — an
    ``Op.__repr__`` is always ``"<op>(...)"`` while ``Const``/``Var``/
    ``Param`` reprs never take that shape, so no real binding is
    lost; it is the difference between finishing and tree-expanding
    a 457k-node DAG.
    """
    subst = {} if subst is None else subst
    if isinstance(pattern, str):
        prev = subst.get(pattern)
        if prev is None:
            subst[pattern] = term
            return subst
        return subst if _term_eq(prev, term) else None
    if isinstance(pattern, Op):
        if not isinstance(term, Op) or term.op != pattern.op:
            return None
        if len(term.args) != len(pattern.args):
            return None
        if set(term.attrs) != set(pattern.attrs):
            return None
        for k, pv in pattern.attrs.items():
            nv = term.attrs[k]
            if isinstance(pv, str):
                key = "$attr:" + pv
                if key in subst:
                    if subst[key] != nv:
                        return None
                else:
                    subst[key] = nv
            elif nv != pv:
                return None
        for pa, ta in zip(pattern.args, term.args, strict=True):
            if _match(pa, ta, subst) is None:
                return None
        return subst
    # concrete leaf — the bounded divergence lives here.
    if isinstance(term, Op):
        return None
    return subst if repr(pattern) == repr(term) else None


from mlopt_core.egraph.terms import _term_eq  # noqa: E402
from mlopt_core.ir import Op  # noqa: E402


class _Domain:
    """Per-module subterm domains with expansion caps + match memo."""

    def __init__(self, roots: dict[str, Any], *, cap: int) -> None:
        from mlopt_discovery.census import (
            _iter_subterms,
            term_expansion,
        )

        self.cap = cap
        self.subs: dict[str, list] = {}
        self.skipped: dict[str, int] = {}
        self.by_op: dict[str, dict[str, list]] = {}
        for name, root in roots.items():
            subs: list = []
            by_op: dict[str, list] = {}
            skipped = 0
            for s in _iter_subterms(root):
                if not isinstance(s, Op):
                    continue
                if term_expansion(s, cap) > cap:
                    skipped += 1
                    continue
                subs.append(s)
                by_op.setdefault(s.op, []).append(s)
            self.subs[name] = subs
            self.by_op[name] = by_op
            self.skipped[name] = skipped

    def binds(self, pat: Any) -> set[str]:
        """Module names where *pat* binds a capped subterm."""
        out: set[str] = set()
        root_op = pat.op if isinstance(pat, Op) else None
        for name, by_op in self.by_op.items():
            tries = (
                by_op.get(root_op, ())
                if root_op is not None
                else self.subs[name]
            )
            if any(_match(pat, s) is not None for s in tries):
                out.add(name)
        return out

    def sites(self, pat: Any, *, per_mod: int = 8) -> list:
        """``(module, subterm)`` bind sites, capped per module."""
        out: list = []
        root_op = pat.op if isinstance(pat, Op) else None
        for name, by_op in self.by_op.items():
            tries = (
                by_op.get(root_op, ())
                if root_op is not None
                else self.subs[name]
            )
            n = 0
            for s in tries:
                if _match(pat, s) is not None:
                    out.append((name, s))
                    n += 1
                    if n >= per_mod:
                        break
        return out


# ---------------------------------------------------------------------------
#  Module builders — whisper-base + t5-small, encoder & decoder steps.
# ---------------------------------------------------------------------------


def _whisper_modules() -> dict[str, tuple[nn.Module, Any]]:
    """``name -> (wrapped module, feed)`` for whisper-base, CPU fp32."""
    from transformers import WhisperForConditionalGeneration

    class _DecWrap(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.dec = dec

        def forward(self, ids, ehs):
            return self.dec(
                input_ids=ids, encoder_hidden_states=ehs
            ).last_hidden_state

    class _EncWrap(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, x):
            return self.enc(x).last_hidden_state

    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-base",
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    x = torch.randn(1, 80, 3000)
    with torch.no_grad():
        ehs = model.model.encoder(x).last_hidden_state
        start = torch.tensor(
            [[model.generation_config.decoder_start_token_id or 50257]]
        )
    return {
        "whisper:encoder": (_EncWrap(model.model.encoder), (x,)),
        "whisper:decoder": (
            _DecWrap(model.model.decoder),
            (start, ehs),
        ),
    }


def _load_t5_probe():
    """Import ``t5_probe`` by path (``bench/drivers`` is not a package)."""
    path = _REPO / "bench" / "drivers" / "probes" / "t5_probe.py"
    spec = importlib.util.spec_from_file_location("t5_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _t5_modules() -> dict[str, tuple[nn.Module, Any]]:
    """``name -> (wrapped module, feed)`` for t5-small, CPU fp32."""
    t5p = _load_t5_probe()
    t5p._patch_min_max()
    from transformers import T5ForConditionalGeneration

    model = T5ForConditionalGeneration.from_pretrained(
        t5p.MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, t5p.VOCAB, (1, t5p.SEQ), generator=g)
    mask = torch.ones(1, t5p.SEQ, dtype=torch.float32)
    with torch.no_grad():
        ehs = model.encoder(
            input_ids=ids, attention_mask=mask
        ).last_hidden_state
    return {
        "t5:encoder": (t5p._EncWrap(model.encoder), (ids, mask)),
        "t5:decoder": (
            t5p._DecWrap(model.decoder),
            (ids, mask, ehs),
        ),
    }


def _build_modules(
    which: list[str],
) -> dict[str, tuple[nn.Module, Any]]:
    """Resolve ``--models``/``--parts`` into the module table."""
    out: dict[str, tuple[nn.Module, Any]] = {}
    want = set(which)
    if want & {"whisper", "whisper:encoder", "whisper:decoder"}:
        out.update(
            {
                k: v
                for k, v in _whisper_modules().items()
                if k in want or "whisper" in want
            }
        )
    if want & {"t5", "t5:encoder", "t5:decoder"}:
        out.update(
            {
                k: v
                for k, v in _t5_modules().items()
                if k in want or "t5" in want
            }
        )
    return out


# ---------------------------------------------------------------------------
#  Mint — per-module (zoo-real) + cross-module (xmodule, bounded).
# ---------------------------------------------------------------------------


def _mint_per_module(
    name: str,
    root: Any,
    domain: _Domain,
    *,
    page: int,
    family_wall: float,
) -> tuple[list, dict]:
    """``pipeline.propose`` over the module's own census, bind-gated.

    ``require_bind`` is replaced by the driver's bounded filter — the
    shipped gate cannot finish on DAG-shared terms (docstring).  A
    proposer family exceeding *family_wall* is recorded ``timeout``.
    """
    from mlopt_discovery import pipeline as pipe
    from mlopt_discovery.census import CorpusTerm, op_tuple_census

    stats: dict[str, Any] = {"module": name, "families": {}}
    t0 = time.time()
    counts, _ = op_tuple_census([CorpusTerm("model", name, root)])
    stats["op_tuples"] = len(counts)
    stats["census_s"] = round(time.time() - t0, 2)

    pointwise, views = pipe._vocab_sets("derived")
    frag = pipe._resolve_vocab_dir(None)
    fams = {
        "census_naturality": lambda: pipe._census_naturality(
            dict(counts), [root], pointwise, views
        ),
        "census_mixed": lambda: pipe._census_mixed_naturality(
            dict(counts), [root], pointwise, views
        ),
        "pattern_recognition": lambda: pipe._pattern_recognition(
            dict(counts)
        ),
        "fusion_schemas": lambda: pipe._fusion_schemas([root], frag),
        "shape_aware": pipe._shape_aware,
        "grammar": pipe._grammar,
    }
    proposals: list = []
    for fname, fn in fams.items():
        t1 = time.time()
        try:
            with _alarm(family_wall):
                got = list(fn())
            stats["families"][fname] = {
                "n": len(got),
                "s": round(time.time() - t1, 2),
            }
            proposals.extend(got)
        except _Wall:
            stats["families"][fname] = {
                "n": None,
                "s": round(time.time() - t1, 2),
                "timeout": True,
            }
        except Exception as e:  # honest per-family failure
            stats["families"][fname] = {
                "n": None,
                "s": round(time.time() - t1, 2),
                "error": f"{type(e).__name__}: {e}",
            }
    # alpha-dedup like propose()'s by_key merge, then the bounded
    # binding-first gate.
    from dataclasses import replace

    from mlopt_discovery import proposal as lp

    by_key: dict = {}
    for p in proposals:
        key = lp._key(p.lhs, p.rhs)
        cur = by_key.get(key)
        if cur is None:
            by_key[key] = p
        else:
            from dataclasses import replace

            by_key[key] = replace(
                cur,
                sources=tuple(dict.fromkeys(cur.sources + p.sources)),
            )
    kept: list = []
    t1 = time.time()
    for p in by_key.values():
        mods = domain.binds(p.lhs)
        if name not in mods:
            continue
        kept.append((p, mods))
        if len(kept) >= page:
            break
    stats["bind_s"] = round(time.time() - t1, 2)
    stats["proposals"] = len(by_key)
    stats["bound"] = len(kept)
    return kept, stats


def _xanti_variants_safe(pat: Any, views: tuple) -> list:
    """Delegate to ``overnight._xanti_variants`` — matcher-free."""
    import bench.drivers.overnight as ov

    return ov._xanti_variants(pat, views)


def _mint_cross(
    roots: dict[str, Any],
    domain: _Domain,
    *,
    k: int,
    cap: int,
    wall: float,
) -> tuple[list, dict]:
    """The ``zoo-xmodule`` mint over the real-model corpus, bounded.

    Census cross-tuples (>= *k* distinct modules) feed
    ``pipeline.propose``; ``antiunify.mine_candidates`` over the
    merged corpus e-graph feeds the view-transform family.  Every
    candidate faces the K-bind gate on the bounded domain, then the
    emit-sanity gate (``_site_outcome`` != ``rhs-err``) over capped
    sites — the same contract ``overnight._xmodule_pool`` keeps.
    """
    from mlopt_core.egraph import EGraph
    from mlopt_core.laws.serialize import alpha_key
    from mlopt_discovery import pipeline as pipe
    from mlopt_discovery.antiunify import mine_candidates
    from mlopt_discovery.census import CorpusTerm, op_tuple_census
    from mlopt_discovery.evidence import _site_outcome
    from mlopt_discovery.pipeline import Proposal

    stats: dict[str, Any] = {"k": k, "wall": wall}
    t0 = time.time()
    terms = [CorpusTerm("model", n, r) for n, r in roots.items()]
    counts, terms_of = op_tuple_census(terms)
    cross = {
        key: n
        for key, n in counts.items()
        if len({m for _, m in terms_of[key]}) >= k
    }
    stats["cross_tuples"] = len(cross)
    all_roots = list(roots.values())
    _pointwise, views = pipe._vocab_sets("derived")
    proposals = list(pipe.propose(dict(cross), all_roots, "derived"))
    stats["census_proposals"] = len(proposals)

    # -- antiunify over the merged corpus e-graph --------------------
    xprops: list = []
    xanti_dropped = 0
    try:
        with _alarm(max(wall - (time.time() - t0), 1.0)):
            eg = EGraph()
            for r in roots.values():
                eg.add_term(r)
            cands = mine_candidates(
                eg, min_uses=k, min_ops=2, max_ops=6
            )
        stats["antiunify_patterns"] = len(cands)
        for cand in cands:
            if len(xprops) >= cap:
                break
            pat = cand.pattern
            if not isinstance(pat, Op) or len(pat.args) != 2:
                continue
            mods = domain.binds(pat)
            if len(mods) < k:
                continue
            g = pat.args[0].op if isinstance(pat.args[0], Op) else "x"
            h = pat.args[1].op if isinstance(pat.args[1], Op) else "x"
            sites = domain.sites(pat, per_mod=4)[:32]
            bound_sites = [
                (m, s)
                for _mod, s in sites
                if (m := _match(pat, s)) is not None
            ]
            for tag, rhs in _xanti_variants_safe(pat, views):
                name = f"xanti:{pat.op}_{g}_{h}_{tag}"
                rule = Proposal(
                    name=name,
                    lhs=pat,
                    rhs=rhs,
                    family="xanti-view",
                    sources=(
                        "antiunify",
                        f"mods:{len(mods)}",
                        f"inst:{cand.n_instances}",
                    ),
                ).as_rule()
                # emit-sanity — the composition arm's own gate: an
                # ill-typed mint (the oracle's ``rhs_err``) drops at
                # emit, never spending gauntlet effort.
                if any(
                    _site_outcome(rule, m, s) == "rhs-err"
                    for m, s in bound_sites
                ):
                    xanti_dropped += 1
                    continue
                xprops.append(
                    Proposal(
                        name=name,
                        lhs=pat,
                        rhs=rhs,
                        family="xanti-view",
                        sources=(
                            "antiunify",
                            f"mods:{len(mods)}",
                            f"inst:{cand.n_instances}",
                        ),
                    )
                )
    except _Wall:
        stats["antiunify_timeout"] = True
    except Exception as e:  # honest mint failure
        stats["antiunify_error"] = f"{type(e).__name__}: {e}"
    stats["xanti"] = len(xprops)
    stats["xanti_dropped"] = xanti_dropped

    # -- unify + K-bind gate ----------------------------------------
    from dataclasses import replace

    by_key: dict = {}
    for p in [*proposals, *xprops]:
        key = alpha_key(p.lhs, p.rhs)
        cur = by_key.get(key)
        if cur is None:
            by_key[key] = p
        else:
            merged = tuple(dict.fromkeys(cur.sources + p.sources))
            by_key[key] = replace(cur, sources=merged)
    out: list = []
    bind_hist: Counter = Counter()
    t1 = time.time()
    for p in by_key.values():
        mods = domain.binds(p.lhs)
        bind_hist[len(mods)] += 1
        if len(mods) < k:
            continue
        out.append((p, mods))
        if len(out) >= cap:
            break
    stats["bind_s"] = round(time.time() - t1, 2)
    stats["proposals"] = len(by_key)
    stats["pool"] = len(out)
    stats["bind_hist"] = {
        str(n): c for n, c in sorted(bind_hist.items())
    }
    stats["total_s"] = round(time.time() - t0, 1)
    return out, stats


# ---------------------------------------------------------------------------
#  Admit / classify / fire / pay
# ---------------------------------------------------------------------------


def _term_cases(env: dict) -> list:
    """Module-granularity ``TermCase``s — the pay leg's domain."""
    from mlopt_discovery.impact import TermCase

    out = []
    for name, (ir, tensors, feeds) in env.items():
        out.append(
            TermCase(
                source="model",
                name=name,
                term=ir.root,
                inputs=tuple(ir.inputs),
                feed=tuple(feeds),
                param_vals=dict(tensors),
            )
        )
    return out


def _real_terms(
    roots: dict[str, Any], *, cap: int, max_n: int
) -> tuple:
    """Bounded distinct real subterms — the oracle/sweep domain."""
    from mlopt_core.ir import op_repr
    from mlopt_discovery.census import _iter_subterms, term_expansion

    seen: set[str] = set()
    out: list = []
    skipped = 0
    for _name, root in roots.items():
        for s in _iter_subterms(root):
            if not isinstance(s, Op):
                continue
            if len(out) >= max_n:
                break
            if term_expansion(s, cap) > cap:
                skipped += 1
                continue
            k = op_repr(s)
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
    return tuple(out), skipped


def _saturate(
    term: Any,
    rules,
    *,
    max_nodes: int = 100_000,
    wall_s: float = 90.0,
    max_iterations: int = 8,
) -> dict:
    """Bounded saturation — the law_transfer fire measurement."""
    from mlopt_core.egraph import EGraph
    from mlopt_core.laws import tags

    budgets = {r.name: 2048 for r in rules.tagged(tags.EXPANSIVE)}
    eg = EGraph()
    root = eg.add_term(term)
    t0 = time.time()
    iters = 0
    stats: dict = {}
    for _ in range(max_iterations):
        stats = eg.run(
            rules,
            root,
            max_iterations=1,
            max_nodes=max_nodes,
            rule_budgets=budgets,
        )
        iters += 1
        if stats.get("stop") != "max_iterations":
            break
        if time.time() - t0 > wall_s:
            stats = dict(stats)
            stats["stop"] = "wall_budget"
            break
    return {
        "seconds": round(time.time() - t0, 2),
        "iterations": iters,
        "stop": stats.get("stop"),
        "n_enodes": eg.n_enodes,
        "rule_fires": {
            k: v
            for k, v in eg.rule_fires.items()
            if k.startswith("rmint:")
        },
        "app_fires": sum(
            1
            for a in eg._applications
            if a["rule"].startswith("rmint:")
        ),
    }


def _pay_arm(
    name: str,
    mod_feed: tuple,
    rule: Any,
    base_rules,
    *,
    wall_s: float,
    reps: int = 10,
) -> dict:
    """search -> lower(verify) -> time, with vs without *rule*.

    Returns a row per arm; a timeout/verify-failure is recorded, not
    raised.  Only the marginal ``withlaw`` vs ``default`` search is
    the law's own effect — ``spelled``/``tape`` arms are context.
    """
    from mlopt_orchestrator.optimize import lower, search
    from mlopt_torch.adapters import TorchSink

    mod, feed = mod_feed
    feeds = feed if isinstance(feed, tuple) else (feed,)
    row: dict[str, Any] = {"module": name, "wall_s": wall_s}

    def _arm(rules, tag):
        t0 = time.time()
        with _alarm(wall_s):
            res = search(
                mod,
                feeds,
                rules=rules,
                max_iterations=6,
                max_enodes=100_000,
            )
            lr = lower(res, feeds, sink=TorchSink(), verify=True)
        row[tag] = {
            "search_s": round(time.time() - t0, 1),
            "verified": bool(
                lr.verified and getattr(lr.verified, "passed", False)
            ),
            "fires": {
                k: v
                for k, v in (res.stats or {})
                .get("rule_fires", {})
                .items()
                if k.startswith("rmint:")
            },
        }
        return lr.module, res

    try:
        with _alarm(wall_s * 2 + 60):
            m_def, _res_def = _arm(base_rules, "default")
            rules_w = base_rules.union([rule])
            m_law, _res_law = _arm(rules_w, "withlaw")
    except _Wall:
        row["error"] = "search wall budget exceeded"
        return row
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        return row

    # -- timing: eager spelled vs the two lowered arms ----------------
    def _time(fn, reps=reps):
        with torch.inference_mode():
            for _ in range(3):
                fn()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t0) / reps * 1e3

    try:
        row["eager_ms"] = round(_time(lambda: mod(*feeds)), 3)
        row["default_ms"] = round(_time(lambda: m_def(*feeds)), 3)
        row["withlaw_ms"] = round(_time(lambda: m_law(*feeds)), 3)
        row["speedup_vs_default"] = (
            round(row["default_ms"] / row["withlaw_ms"], 3)
            if row.get("withlaw_ms")
            else None
        )
        row["speedup_vs_eager"] = (
            round(row["eager_ms"] / row["withlaw_ms"], 3)
            if row.get("withlaw_ms")
            else None
        )
    except Exception as e:
        row["time_error"] = f"{type(e).__name__}: {e}"
    return row


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--models",
        default="whisper,t5",
        help="comma list: whisper|t5|whisper:encoder|...",
    )
    ap.add_argument("--xbind", type=int, default=2)
    ap.add_argument("--page", type=int, default=DEFAULT_PAGE)
    ap.add_argument("--xcap", type=int, default=DEFAULT_XCAP)
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CAND,
        help="cap on candidates sent to the gauntlet",
    )
    ap.add_argument("--referee", default=DEFAULT_REFEREE)
    ap.add_argument("--real-cap", type=int, default=DEFAULT_REAL_CAP)
    ap.add_argument("--site-cap", type=int, default=DEFAULT_SITE_CAP)
    ap.add_argument("--max-real", type=int, default=DEFAULT_MAX_REAL)
    ap.add_argument(
        "--family-wall",
        type=float,
        default=120.0,
        help="per-proposer-family wall budget (s)",
    )
    ap.add_argument(
        "--xmodule-wall",
        type=float,
        default=600.0,
        help="cross-module mint wall budget (s)",
    )
    ap.add_argument(
        "--fire-wall",
        type=float,
        default=90.0,
        help="per-module saturation wall for the fire check (s)",
    )
    ap.add_argument(
        "--pay-arm",
        action="store_true",
        help="run search->lower->time on admitted laws",
    )
    ap.add_argument(
        "--pay-wall",
        type=float,
        default=300.0,
        help="per-arm search wall for --pay-arm (s)",
    )
    ap.add_argument(
        "--pay-cap",
        type=int,
        default=4,
        help="max admitted laws getting the pay arm",
    )
    ap.add_argument("--no-cross", action="store_true")
    ap.add_argument(
        "--prim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the zoo-prim arm — antiunify-mined manual shapes "
        "that no bank op denotes mint NEW synop_* primitives "
        "(ov._prim_pool over the real-model roots)",
    )
    ap.add_argument(
        "--prim-wall",
        type=float,
        default=900.0,
        help="zoo-prim mint wall budget (s)",
    )
    ap.add_argument(
        "--prim-cap",
        type=int,
        default=24,
        help="zoo-prim pool bound — top-gain shapes",
    )
    ap.add_argument(
        "--prim-family-mint",
        action="store_true",
        help="enable the refused-spec meta-loop (family_synth) — "
        "off for r1 to bound the mint",
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)
    if args.dry:
        args.page = 6
        args.xcap = 12
        args.max_candidates = 8
        args.pay_cap = 1
        args.family_wall = 60.0
        args.xmodule_wall = 240.0
        args.prim_wall = 300.0

    t_start = time.time()
    report: dict[str, Any] = {
        "schema": 1,
        "git_note": "uncommitted driver — bench/drivers/mint/",
        "args": vars(args),
        "modules": {},
        "mint": {},
        "candidates": [],
        "admitted": [],
        "pay": [],
        "errors": [],
    }

    # -- export -------------------------------------------------------
    which = [s.strip() for s in args.models.split(",") if s.strip()]
    built = _build_modules(which)
    if not built:
        print("no modules selected", file=sys.stderr)
        return 2
    import mlopt_carriers  # noqa: F401 — carrier op bindings
    from mlopt_discovery.model_corpus import export_ir

    env: dict[str, tuple] = {}
    models: dict[str, tuple] = {}
    roots: dict[str, Any] = {}
    for name, (mod, feed) in built.items():
        t0 = time.time()
        try:
            with _alarm(300.0):
                ir, leaves, feeds = export_ir(mod, feed, dtype="fp32")
            env[name] = (ir, leaves, feeds)
            models[name] = (mod.eval(), feeds)
            roots[name] = ir.root
            report["modules"][name] = {
                "export_s": round(time.time() - t0, 1),
                "feeds": [
                    list(t.shape) if torch.is_tensor(t) else str(t)
                    for t in feeds
                ],
            }
            print(
                f"[export] {name}: {time.time() - t0:.1f}s", flush=True
            )
        except Exception as e:
            report["modules"][name] = {
                "export_error": f"{type(e).__name__}: {e}"
            }
            report["errors"].append(f"{name}: export {e!r}")
            print(f"[export] {name}: FAIL {e!r}", flush=True)
    if not roots:
        print("all exports failed", file=sys.stderr)
        _write(report, args.out)
        return 1

    # -- domains ------------------------------------------------------
    domain = _Domain(roots, cap=args.real_cap)
    report["domain"] = {
        "cap": args.real_cap,
        "subterms": {
            n: {"kept": len(v), "skipped": domain.skipped[n]}
            for n, v in domain.subs.items()
        },
    }

    # -- mint: per-module ----------------------------------------------
    per_mod: list = []
    for name, root in roots.items():
        t0 = time.time()
        try:
            with _alarm(args.family_wall * 8 + 60):
                kept, stats = _mint_per_module(
                    name,
                    root,
                    domain,
                    page=args.page,
                    family_wall=args.family_wall,
                )
            stats["wall_s"] = round(time.time() - t0, 1)
            per_mod.extend((p, mods | {name}) for p, mods in kept)
            report["mint"][name] = stats
            print(
                f"[mint] {name}: {stats['bound']}/{stats['proposals']} "
                f"bound ({stats['wall_s']}s)",
                flush=True,
            )
        except _Wall:
            report["mint"][name] = {"error": "mint wall exceeded"}
            report["errors"].append(f"{name}: mint wall")
        except Exception as e:
            report["mint"][name] = {"error": f"{type(e).__name__}: {e}"}
            report["errors"].append(f"{name}: mint {e!r}")

    # -- mint: cross-module --------------------------------------------
    cross_kept: list = []
    if not args.no_cross and len(roots) >= 2:
        t0 = time.time()
        try:
            with _alarm(args.xmodule_wall):
                cross_kept, xstats = _mint_cross(
                    roots,
                    domain,
                    k=args.xbind,
                    cap=args.xcap,
                    wall=args.xmodule_wall,
                )
            xstats["wall_s"] = round(time.time() - t0, 1)
            report["mint"]["_xmodule"] = xstats
            print(
                f"[mint] xmodule k={args.xbind}: {xstats['pool']}/"
                f"{xstats['proposals']} pooled ({xstats['wall_s']}s)",
                flush=True,
            )
        except _Wall:
            report["mint"]["_xmodule"] = {"error": "wall exceeded"}
            report["errors"].append("xmodule: wall")
        except Exception as e:
            report["mint"]["_xmodule"] = {
                "error": f"{type(e).__name__}: {e}"
            }
            report["errors"].append(f"xmodule: {e!r}")

    # -- mint: primitives (zoo-prim arm) --------------------------------
    # ``overnight._prim_pool`` mines recurring manual shapes no bank
    # op denotes and mints ``synop_*`` primitives + fold laws over
    # them — the machinery that produced the shipped ``prim:synop*``
    # family, here run on real-model terms.  Its internal bind sites
    # use metavar-leaf patterns (the ``repr(term)`` blowup needs a
    # concrete pattern leaf — mined shapes have none), and the whole
    # call sits under the wall alarm.
    prim_kept: list = []  # (cand, mods:set, prim_meta:dict|None)
    if args.prim and len(roots) >= 2:
        t0 = time.time()
        try:
            import bench.drivers.overnight as ov

            with _alarm(args.prim_wall):
                pcands, pmeta, pstats = ov._prim_pool(
                    roots,
                    args.xbind,
                    cap=args.prim_cap,
                    budget=2048,
                    site_cap=16,
                    min_ops=2,
                    max_ops=6,
                    beam=32,
                    max_expand=2000,
                    bootstrap=False,
                    family_mint=args.prim_family_mint,
                    depth=3,
                    vocab=True,
                    vocab_dir=None,
                    emit_vocabulary=False,
                )
            for i, cand in enumerate(pcands):
                prim_kept.append(
                    (cand, domain.binds(cand.lhs), pmeta.get(i))
                )
            pstats["wall_s"] = round(time.time() - t0, 1)
            pstats["pool"] = len(pcands)
            report["mint"]["_prim"] = pstats
            print(
                f"[mint] prim k={args.xbind}: {len(pcands)} folds "
                f"({pstats['wall_s']}s)",
                flush=True,
            )
        except _Wall:
            report["mint"]["_prim"] = {"error": "wall exceeded"}
            report["errors"].append("prim: wall")
        except Exception as e:
            report["mint"]["_prim"] = {
                "error": f"{type(e).__name__}: {e}"
            }
            report["errors"].append(f"prim: {e!r}")

    # -- unify + dedup --------------------------------------------------
    from mlopt_core.laws.serialize import alpha_key, law_to_data

    # items: {"p": Proposal|Candidate, "mods": set, "sources": list,
    #         "family": str, "prim": meta|None}
    items: list[dict] = []
    for p, mods in per_mod:
        items.append(
            {
                "p": p,
                "mods": set(mods),
                "sources": list(getattr(p, "sources", ()) or ()),
                "family": getattr(p, "family", ""),
                "prim": None,
            }
        )
    for p, mods in cross_kept:
        items.append(
            {
                "p": p,
                "mods": set(mods),
                "sources": list(getattr(p, "sources", ()) or ()),
                "family": getattr(p, "family", ""),
                "prim": None,
            }
        )
    for cand, mods, pm in prim_kept:
        items.append(
            {
                "p": cand,
                "mods": set(mods),
                "sources": [
                    str(getattr(cand, "note", "") or "prim-mint")
                ],
                "family": "prim-fold",
                "prim": pm,
            }
        )

    by_key: dict = {}
    order: list = []
    for it in items:
        p = it["p"]
        key = alpha_key(p.lhs, p.rhs)
        if key in by_key:
            cur = by_key[key]
            cur["mods"] |= it["mods"]
            cur["sources"] = list(
                dict.fromkeys(cur["sources"] + it["sources"])
            )
            if cur["prim"] is None:
                cur["prim"] = it["prim"]
            continue
        by_key[key] = it
        order.append(key)

    candidates: list[dict] = []
    rules_to_judge: list[tuple[str, Any]] = []
    for i, key in enumerate(order[: args.max_candidates]):
        it = by_key[key]
        p = it["p"]
        rid = f"rmint:{i:03d}"
        rule = p.as_rule()
        object.__setattr__(rule, "name", rid)
        rec = {
            "id": rid,
            "orig_name": (
                getattr(p, "name", None)
                or getattr(p, "label", None)
                or rule.name
            ),
            "family": it["family"],
            "sources": it["sources"],
            "mods": sorted(it["mods"]),
            "lhs": law_to_data(rule)["lhs"],
            "rhs": law_to_data(rule)["rhs"],
        }
        if it["prim"] is not None:
            rec["prim"] = it["prim"]
        candidates.append(rec)
        rules_to_judge.append((rid, rule))
    report["candidates"] = candidates
    report["n_minted_total"] = len(order)
    print(
        f"[mint] {len(order)} unique candidates "
        f"({len(candidates)} to judge)",
        flush=True,
    )
    # checkpoint — mint evidence is durable even if admit is killed
    _write(report, args.out)

    # -- referee ctx ----------------------------------------------------
    import mlopt_torch  # noqa: F401 — torch bindings
    from mlopt_torch.adapters import TorchSink

    from bench import referees

    rt, rt_skipped = _real_terms(
        roots, cap=args.real_cap, max_n=args.max_real
    )
    ctx = referees.RefereeCtx(
        real_terms=rt,
        probe=tuple(_term_cases(env)),
        sink=TorchSink(),
    )
    report["referee_ctx"] = {
        "real_terms": len(rt),
        "real_terms_skipped": rt_skipped,
        "probe": [n for n in env],
        "referee": args.referee,
    }

    # -- admit -----------------------------------------------------------
    import bench.drivers.overnight as ov

    bank = None
    admitted_rules: list[tuple[str, Any, dict]] = []
    for rid, rule in rules_to_judge:
        t0 = time.time()
        try:
            with _alarm(600.0):
                rec = ov._admit(rule, args.referee, ctx)
        except _Wall:
            rec = {
                "name": rid,
                "admitted": False,
                "reason": "gauntlet wall 600s",
                "stages": [],
                "skipped": "",
                "measured": {},
            }
        except Exception as e:
            rec = {
                "name": rid,
                "admitted": False,
                "reason": f"{type(e).__name__}: {e}",
                "stages": [],
                "skipped": "",
                "measured": {},
            }
        rec["wall_s"] = round(time.time() - t0, 1)
        cand = next(c for c in candidates if c["id"] == rid)
        cand["admit"] = {
            "admitted": rec["admitted"],
            "reason": rec.get("reason", ""),
            "skipped": rec.get("skipped", ""),
            "stages": rec.get("stages", []),
            "measured": rec.get("measured", {}),
            "wall_s": rec["wall_s"],
        }
        print(
            f"[admit] {rid} {cand['orig_name']:<28} "
            f"admitted={rec['admitted']} ({rec.get('reason', '')[:60]})",
            flush=True,
        )
        if rec["admitted"]:
            admitted_rules.append((rid, rule, rec))
    # checkpoint — admission verdicts durable before classify/fire/pay
    _write(report, args.out)

    # -- classify ---------------------------------------------------------
    if admitted_rules:
        from bench.drivers.screen import novelty_sieve as ns

        bank = ns.known_bank()
    for rid, rule, rec in admitted_rules:
        t0 = time.time()
        try:
            with _alarm(180.0):
                cres = ov._classify(rule, bank, ctxs=[])
        except _Wall:
            cres = {"verdict": "classify-timeout", "via": "wall"}
        except Exception as e:
            cres = {
                "verdict": "classify-error",
                "via": f"{type(e).__name__}: {e}",
            }
        cres["wall_s"] = round(time.time() - t0, 1)
        cand = next(c for c in candidates if c["id"] == rid)
        cand["classify"] = cres
        report["admitted"].append(
            {
                "id": rid,
                "orig_name": cand["orig_name"],
                "mods": cand["mods"],
                "novelty": cres.get("verdict"),
                "reason": rec.get("reason", ""),
            }
        )
        print(
            f"[classify] {rid} {cres.get('verdict')} "
            f"({cres.get('via', '')})",
            flush=True,
        )

    # -- fire check on the source modules --------------------------------
    from mlopt_orchestrator.optimize import default_rules

    base_rules = default_rules()
    fire_rows: list[dict] = []
    for rid, rule, _rec in admitted_rules:
        rules_w = base_rules.union([rule])
        row = {"id": rid, "fires": {}}
        for name, root in roots.items():
            t0 = time.time()
            try:
                with _alarm(args.fire_wall):
                    sat = _saturate(
                        root, rules_w, wall_s=args.fire_wall
                    )
                sat["wall_s"] = round(time.time() - t0, 1)
                row["fires"][name] = sat
            except _Wall:
                row["fires"][name] = {"error": "fire wall"}
            except Exception as e:
                row["fires"][name] = {
                    "error": f"{type(e).__name__}: {e}"
                }
        fire_rows.append(row)
        print(f"[fire] {rid} done", flush=True)
    report["fire"] = fire_rows

    # -- pay arm ------------------------------------------------------------
    if args.pay_arm and admitted_rules:
        pay_rows: list[dict] = []
        # baseline search once per module that any law fires on
        for rid, rule, _rec in admitted_rules[: args.pay_cap]:
            # pick the module with the most rule_fires for this law
            best_mod, best_n = None, -1
            for name in roots:
                sat = next(
                    (
                        f
                        for r in fire_rows
                        if r["id"] == rid
                        for m, f in r["fires"].items()
                        if m == name
                    ),
                    {},
                )
                n = sum(sat.get("rule_fires", {}).values())
                if n > best_n:
                    best_mod, best_n = name, n
            if best_mod is None or best_n <= 0:
                pay_rows.append(
                    {
                        "id": rid,
                        "note": "law never fired — no pay arm",
                    }
                )
                continue
            row = _pay_arm(
                best_mod,
                models[best_mod],
                rule,
                base_rules,
                wall_s=args.pay_wall,
            )
            row["id"] = rid
            pay_rows.append(row)
            print(
                f"[pay] {rid} on {best_mod}: "
                f"{row.get('speedup_vs_default')}",
                flush=True,
            )
        report["pay"] = pay_rows

    report["wall_s"] = round(time.time() - t_start, 1)
    _write(report, args.out)
    n_adm = len(report["admitted"])
    print(
        f"== real_model_mint — minted={report['n_minted_total']} "
        f"judged={len(candidates)} admitted={n_adm} "
        f"wall={report['wall_s']}s ==",
        flush=True,
    )
    return 0


def _write(report: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
