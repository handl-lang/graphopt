"""Law-transfer probe — do bank laws fire off their minting corpus?

The law bank mixes hand laws with 58 machine-minted ``prim:synop*``
folds (``mlopt_core.laws.minted`` — the ``zoo-prim`` / bootstrap /
vocab-priced mints, plus the ``AUTO_TIER`` auto-admission).  Every
prim record carries ``mods`` — the zoo modules whose op-patterns the
mint saw.  Nobody has measured the *cross-model* question: does a law
minted on ``GatedLinearAttention``/``MambaSelectiveScan`` shapes
actually fire on a whisper decoder graph, a ``ChunkedRetention``
module, or a ``model_corpus`` mutant?

This driver saturates a fixed module set under the *shipped*
``default_rules()`` composition (core ``DEFAULT`` + the carrier
preset — the same rule list ``Optimizer.search`` resolves for
``rules=None``) and records, per module:

* ``rule_fires`` — ``EGraph.rule_fires`` (merge-count per rule),
  plus ``app_fires`` — per-rule counts over ``eg._applications``
  (every witnessed firing, merge or fresh-instantiation — a law can
  fire without merging when its RHS is already in the class);
* the op census and the ``(op, child-ops)`` tuple census, for the
  coverage-gap analysis;
* wall time, iterations, stop reason, enode/class counts.

Saturation mirrors the shipped bounds: ``symmetry_budget=2048`` per
``EXPANSIVE``-tagged rule (``optimize.search``'s default),
``max_nodes`` cap, and a per-module wall budget checked between
one-iteration ``eg.run`` calls (the frontier/delta state persists
across calls — same trick ``decoder_search_probe`` uses).

Module set (``--only`` filters by substring):

* **zoo mint-corpus** — ``GatedLinearAttention``,
  ``MambaSelectiveScan``, ``RealGatedLRU``, ``GatedDeltaRule`` (the
  modules named in nearly every prim record's ``mods``);
* **zoo off-corpus** — ``ChunkedRetention``, ``RMSNormPerHead``,
  ``ConvNeXtStage``, ``GEGLUBlock``, ``RotaryPair``, ``DiffAttention``
  (zoo modules absent from every prim record's ``mods``);
* **whisper** — the whisper-base decoder step and encoder frontend,
  CPU fp32 (``decoder_search_probe``'s export path), plus the five
  whisper-shaped fragments ``kernelopt_bridge_probe`` spells by hand
  (``attn_core`` / ``mlp_block`` / ``conv_block`` / ``o_proj`` /
  ``ln_res`` — synthetic re-spellings, tagged ``frag:``);
* **mutants** — ``model_corpus.generate_mutants`` over
  ``ChunkedRetention``'s exported term, donor stats mined over the
  probe's own exported TermCases (the same machinery
  ``model_corpus`` runs, on a corpus small enough to stay cheap).

Honesty contract: a module that fails export or saturates to its
wall/node budget is recorded as a row with the failure/bound noted —
never silently dropped.  "Fired" means *merged into the e-graph*
(``rule_fires``); ``app_fires`` is the wider witnessed-firing count.
A law that never fires anywhere is dead weight *for this module
set* — reported, not condemned.

Run::

    .venv/bin/python bench/drivers/probes/law_transfer_probe.py
    .venv/bin/python bench/drivers/probes/law_transfer_probe.py --only zoo --out /tmp/lt.json
    .venv/bin/python bench/drivers/probes/law_transfer_probe.py --no-whisper --no-mutants
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import mlopt_carriers  # noqa: F401 — registers the CARRIERS preset
import mlopt_torch  # noqa: F401 — torch bindings for export
import torch
import torch.nn as nn
from mlopt_core.egraph import EGraph
from mlopt_core.ir import Op
from mlopt_core.laws import tags
from mlopt_core.laws.provenance import (
    AUTO_ADMITTED,
    MACHINE_SHIPPED,
    MACHINE_STORED,
    law_provenance,
)
from mlopt_discovery.model_corpus import (
    _term_case,
    export_ir,
    generate_mutants,
)
from mlopt_discovery.zoo import zoo as zoo_registry
from mlopt_orchestrator.optimize import default_rules
from mlopt_torch.adapters import TorchSource

REPO = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO / "docs" / "reports" / "law_transfer.json"

#: Per-rule enode budget for EXPANSIVE-tagged rules — the shipped
#: ``optimize.search`` ``symmetry_budget`` default.
SYMMETRY_BUDGET = 2048
DEFAULT_MAX_NODES = 100_000  # ``search``'s ``max_enodes`` default
DEFAULT_WALL_S = 120.0

#: Zoo modules named in the prim records' ``mods`` — the minting
#: corpus itself.
ZOO_MINT_CORPUS = [
    "GatedLinearAttention",
    "MambaSelectiveScan",
    "RealGatedLRU",
    "GatedDeltaRule",
    "BlockSparseAttention",
    "SlidingWindowAttention",
    "RWKVMixBlock",
    "DecoupledMLA",
]

#: Zoo modules absent from every prim record's ``mods`` — the
#: off-corpus half of the transfer question.
ZOO_OFF_CORPUS = [
    "ChunkedRetention",
    "RMSNormPerHead",
    "ConvNeXtStage",
    "GEGLUBlock",
    "RotaryPair",
    "DiffAttention",
    "AdditiveAttention",
    "TalkingHeadsAttention",
    "SigmoidAttention",
    "SqueezeExcite",
]

WHISPER_MODEL = "openai/whisper-base"
WHISPER_N_MELS = 80
WHISPER_N_FRAMES = 3000  # whisper requires exactly 30 s of log-mel

#: Mutant-generation bounds — ``key_cap`` caps tree-expansion for the
#: novelty ``shape_key``s (the whisper exports' shared DAGs expand
#: exponentially unbounded); ``_MUTANT_HEADROOM`` is the node budget
#: mutants may grow past the base term's size (``model_corpus``'s
#: ``mutant_headroom`` convention).
_MUTANT_KEY_CAP = 20_000
_MUTANT_HEADROOM = 60

#: Mutant bases — one off-corpus retention module and one off-corpus
#: attention module, so the generated probes sit on both sides of
#: the minting corpus's family boundary.
_MUTANT_BASES = ["ChunkedRetention", "DiffAttention"]


# ---------------------------------------------------------------------------
#  Module builders — zoo workloads, whisper subgraphs, hand-spelled
#  whisper fragments (the kernelopt_bridge spellings, copied so this
#  probe stays standalone).
# ---------------------------------------------------------------------------


class _DecWrap(nn.Module):
    """Decoder step -> logits only (``whisper_probe``/``decoder_search_probe``)."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, ids, ehs):
        return self.dec(
            input_ids=ids, encoder_hidden_states=ehs
        ).last_hidden_state


class _EncWrap(nn.Module):
    """Encoder -> last_hidden_state (same boundary ``_DecWrap`` uses)."""

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x):
        return self.enc(x).last_hidden_state


class _AttnCore(nn.Module):
    """``softmax((q @ k^T) * scale) @ v`` — whisper attention fragment."""

    def __init__(self, heads=8, dim=64, scale=None):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.scale = scale if scale is not None else dim**-0.5

    def forward(self, q, k, v):
        b, t, _ = q.shape
        h, d = self.heads, self.dim
        q = q.view(b, t, h, d).transpose(1, 2)
        k = k.view(b, t, h, d).transpose(1, 2)
        v = v.view(b, t, h, d).transpose(1, 2)
        s = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, v)
        return o.transpose(1, 2).reshape(b, t, h * d)


class _MlpBlock(nn.Module):
    """``x + fc2(gelu(fc1(layer_norm(x))))`` — whisper MLP + residual."""

    def __init__(self, d=512, ffn=2048):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)

    def forward(self, x):
        return x + self.fc2(
            torch.nn.functional.gelu(self.fc1(self.ln(x)))
        )


class _ConvBlock(nn.Module):
    """conv1d -> gelu -> conv1d — the whisper encoder frontend pair."""

    def __init__(self, cin=80, mid=512, cout=512):
        super().__init__()
        self.c1 = nn.Conv1d(cin, mid, 3, stride=1, padding=1)
        self.c2 = nn.Conv1d(mid, cout, 3, stride=2, padding=1)

    def forward(self, x):
        return self.c2(torch.nn.functional.gelu(self.c1(x)))


class _OProj(nn.Module):
    """``x + linear(x)`` — residual output projection."""

    def __init__(self, d=512):
        super().__init__()
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        return x + self.proj(x)


class _LnRes(nn.Module):
    """``x + layer_norm(x)`` — bare norm + residual fragment."""

    def __init__(self, d=512):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        return x + self.ln(x)


def _fragment_cases() -> list[tuple[str, nn.Module, tuple]]:
    """The five whisper-shaped fragments at whisper-base dimensions."""
    d, ffn, h, hd, t = 512, 2048, 8, 64, 512
    return [
        (
            "frag:attn_core",
            _AttnCore(h, hd),
            (
                torch.randn(1, t, h * hd),
                torch.randn(1, t, h * hd),
                torch.randn(1, t, h * hd),
            ),
        ),
        (
            "frag:mlp_block",
            _MlpBlock(d, ffn),
            (torch.randn(1, t, d),),
        ),
        (
            "frag:conv_block",
            _ConvBlock(),
            (torch.randn(1, 80, 1000),),
        ),
        (
            "frag:o_proj",
            _OProj(d),
            (torch.randn(1, t, d),),
        ),
        (
            "frag:ln_res",
            _LnRes(d),
            (torch.randn(1, t, d),),
        ),
    ]


# ---------------------------------------------------------------------------
#  Saturation — the shipped bounds, instrumented for fires.
# ---------------------------------------------------------------------------


def _saturate(
    term: Op,
    rules,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    wall_s: float = DEFAULT_WALL_S,
    max_iterations: int = 12,
) -> dict:
    """Saturate *term* under *rules*; return the fire record.

    One-iteration ``eg.run`` calls with a wall-clock check between
    them — the dirty frontier, applied-rule set and rule budgets all
    persist across calls, so the fixed point (or the bound) is
    exactly what a single ``run`` would reach, minus the overrun.
    """
    budgets = {
        r.name: SYMMETRY_BUDGET for r in rules.tagged(tags.EXPANSIVE)
    }
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
    apps: Counter = Counter()
    for app in eg._applications:
        apps[app["rule"]] += 1
    return {
        "seconds": round(time.time() - t0, 2),
        "iterations": iters,
        "stop": stats.get("stop"),
        "n_enodes": eg.n_enodes,
        "n_classes": eg.n_classes,
        "rule_fires": dict(eg.rule_fires),
        "app_fires": dict(apps),
    }


# ---------------------------------------------------------------------------
#  Census helpers
# ---------------------------------------------------------------------------


def _op_census(term: Op) -> Counter:
    """Count ops over the (DAG-shared) term — dedup by object id."""
    counts: Counter = Counter()
    seen: set = set()

    def walk(t):
        if not isinstance(t, Op) or id(t) in seen:
            return
        seen.add(id(t))
        counts[t.op] += 1
        for a in t.args:
            walk(a)

    walk(term)
    return counts


def _pattern_ops(pat, counts=None) -> Counter:
    """Count ops in a rule LHS/RHS pattern (strings are metavars)."""
    if counts is None:
        counts = Counter()
    if isinstance(pat, Op):
        counts[pat.op] += 1
        for a in pat.args:
            _pattern_ops(a, counts)
    return counts


# ---------------------------------------------------------------------------
#  Module runners — each returns (row, term_case_or_None)
# ---------------------------------------------------------------------------


def _run_named(name: str, term: Op, rules, wall_s: float) -> dict:
    row = _saturate(term, rules, wall_s=wall_s)
    row["module"] = name
    return row


def _zoo_term(name: str) -> tuple[Op, object]:
    """Build + export one zoo workload; return ``(term, TermCase)``."""
    wl = next(w for w in zoo_registry(16, 8, 2) if w.name == name)
    model, feed = wl.build()
    ir, leaves, feeds = export_ir(model, feed, dtype="fp32")
    tc = _term_case(f"zoo:{name}", ir, leaves, feeds)
    return tc.term, tc


def _whisper_terms() -> list[tuple[str, Op, object]]:
    """Export the whisper-base decoder step and encoder (CPU fp32).

    Mirrors ``decoder_search_probe.export_ir`` — the eager-attention
    decoder wrapped to return ``last_hidden_state`` so
    ``torch.export`` sees a plain tensor.
    """
    from transformers import WhisperForConditionalGeneration

    out: list[tuple[str, Op, object]] = []
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER_MODEL,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    x = torch.randn(1, WHISPER_N_MELS, WHISPER_N_FRAMES)
    with torch.no_grad():
        ehs = model.model.encoder(x).last_hidden_state
        start = torch.tensor(
            [[model.generation_config.decoder_start_token_id or 50257]]
        )
    for name, mod, feed in (
        (
            "whisper:decoder",
            _DecWrap(model.model.decoder),
            (start, ehs),
        ),
        ("whisper:encoder", _EncWrap(model.model.encoder), (x,)),
    ):
        try:
            ir, leaves = TorchSource().to_ir(mod, feed)
        except Exception as exc:  # honest export-decline row
            out.append((name, None, {"export_error": repr(exc)}))
            continue
        feeds = feed if isinstance(feed, tuple) else (feed,)
        tc = _term_case(name, ir, dict(leaves), tuple(feeds))
        out.append((name, tc.term, tc))
    return out


# ---------------------------------------------------------------------------
#  Analysis — transfer matrix + dead-weight + coverage gaps
# ---------------------------------------------------------------------------


def _rule_class(name: str) -> str:
    """Provenance bucket for one rule name."""
    if name in AUTO_ADMITTED:
        return "auto-admitted"
    if name in MACHINE_SHIPPED:
        return "machine-shipped"
    if name in MACHINE_STORED:
        return "machine-stored"
    return law_provenance(name)


def _prim_mint_corpus() -> dict[str, list[str]]:
    """``prim:<synop>`` fold name -> the record's ``mods`` minting set."""
    from mlopt_core.laws import minted

    out: dict[str, list[str]] = {}
    for synop, rec in minted.prim_records().items():
        mods = rec.get("mods")
        out[f"prim:{synop}"] = list(mods) if mods else []
    return out


def analyze(rows: list[dict], rules) -> dict:
    """Fold the per-module rows into the transfer report."""
    modules = [r["module"] for r in rows]
    fires: dict[str, dict[str, int]] = {}
    apps: dict[str, dict[str, int]] = {}
    for r in rows:
        fires[r["module"]] = r.get("rule_fires", {})
        apps[r["module"]] = r.get("app_fires", {})
    mint_corpus = _prim_mint_corpus()
    rule_names = [r.name for r in rules]
    lhs_ops = {r.name: _pattern_ops(r.lhs) for r in rules}

    # -- per-rule transfer row ------------------------------------
    # A fired module counts as *on-corpus* for a prim fold when its
    # base name (``zoo:GatedLinearAttention`` -> ``GatedLinearAttention``)
    # appears in the record's ``mods``; whisper/frag/mutant modules
    # are off-corpus by construction.
    def _on_corpus(module: str, minted_on: list[str]) -> bool:
        base = module.split(":", 1)[-1]
        return base in minted_on

    per_rule = []
    for name in rule_names:
        fired_on = sorted(m for m, f in fires.items() if f.get(name))
        minted_on = mint_corpus.get(name, [])
        off_corpus = [
            m for m in fired_on if not _on_corpus(m, minted_on)
        ]
        per_rule.append(
            {
                "rule": name,
                "class": _rule_class(name),
                "fired_on": fired_on,
                "n_fired": len(fired_on),
                "total_fires": sum(
                    f.get(name, 0) for f in fires.values()
                ),
                "total_apps": sum(
                    a.get(name, 0) for a in apps.values()
                ),
                "minted_on": minted_on,
                "off_corpus_fires": off_corpus,
                "lhs_ops": dict(lhs_ops[name]),
            }
        )

    # -- prim-fold transfer summary --------------------------------
    prim_rows = [p for p in per_rule if p["rule"].startswith("prim:")]
    prim_summary = {
        "n_prim_rules": len(prim_rows),
        "fired_anywhere": sum(1 for p in prim_rows if p["n_fired"]),
        "never_fired": [
            p["rule"] for p in prim_rows if not p["n_fired"]
        ],
        "off_corpus_firers": [
            {
                "rule": p["rule"],
                "minted_on": p["minted_on"],
                "fired_on": p["fired_on"],
                "off_corpus_fires": p["off_corpus_fires"],
            }
            for p in prim_rows
            if p["off_corpus_fires"]
        ],
    }

    # -- dead weight: bank rules that never fired ------------------
    never = [
        {
            "rule": n,
            "class": _rule_class(n),
            "lhs_ops": dict(lhs_ops[n]),
        }
        for n in rule_names
        if not any(f.get(n) for f in fires.values())
    ]
    dead_machine = [d for d in never if d["class"] != "human"]

    # -- coverage gaps --------------------------------------------
    # ops present in a module but on NO shipped rule LHS; and
    # (op, child-ops) tuples unique to non-zoo modules.
    bank_ops = Counter()
    for ops in lhs_ops.values():
        bank_ops.update(ops)
    gap_rows = []
    for r in rows:
        census = r.get("op_census", {})
        uncovered = {
            op: c for op, c in census.items() if op not in bank_ops
        }
        gap_rows.append(
            {
                "module": r["module"],
                "uncovered_ops": uncovered,
                "n_op_kinds": len(census),
            }
        )

    return {
        "modules": modules,
        "per_rule": per_rule,
        "prim_summary": prim_summary,
        "never_fired": never,
        "dead_machine": dead_machine,
        "coverage_gaps": gap_rows,
        "bank_op_kinds": len(bank_ops),
    }


def _encode_key(key) -> str:
    op, children = key
    return f"{op}({','.join(children)})"


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only", default="", help="substring filter on module name"
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--wall", type=float, default=DEFAULT_WALL_S)
    ap.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    ap.add_argument("--no-whisper", action="store_true")
    ap.add_argument("--no-mutants", action="store_true")
    ap.add_argument("--no-fragments", action="store_true")
    ap.add_argument("--mutants", type=int, default=2)
    ap.add_argument("--mutant-seed", type=int, default=0)
    args = ap.parse_args()

    rules = default_rules()
    print(f"[rules] {len(rules)} rules", flush=True)

    jobs: list[tuple[str, str, float]] = []  # (kind, name, wall)
    for n in ZOO_MINT_CORPUS + ZOO_OFF_CORPUS:
        jobs.append(("zoo", n, args.wall))
    if not args.no_whisper:
        jobs.append(("whisper", "whisper:decoder", args.wall * 2))
        jobs.append(("whisper", "whisper:encoder", args.wall * 2))
    if not args.no_fragments:
        for n, _m, _f in _fragment_cases():
            jobs.append(("frag", n, args.wall))

    rows: list[dict] = []
    term_cases: dict[str, object] = {}

    def keep(name: str) -> bool:
        return not args.only or args.only in name

    # -- zoo -------------------------------------------------------
    for kind, name, wall in jobs:
        display = f"{kind}:{name}" if kind != "whisper" else name
        if kind != "zoo" or not keep(display):
            continue
        try:
            term, tc = _zoo_term(name)
        except Exception as exc:
            rows.append(
                {
                    "module": f"zoo:{name}",
                    "kind": kind,
                    "error": f"export: {type(exc).__name__}: {exc}",
                    "rule_fires": {},
                    "app_fires": {},
                }
            )
            print(f"[zoo:{name}] export FAILED: {exc}", flush=True)
            continue
        term_cases[f"zoo:{name}"] = tc
        row = _run_named(f"zoo:{name}", term, rules, wall)
        row["kind"] = kind
        row["op_census"] = dict(_op_census(term))
        rows.append(row)
        print(
            f"[zoo:{name}] {row['seconds']}s iters={row['iterations']} "
            f"stop={row['stop']} enodes={row['n_enodes']} "
            f"fires={len(row['rule_fires'])}",
            flush=True,
        )

    # -- whisper ---------------------------------------------------
    if not args.no_whisper and (
        not args.only or "whisper" in args.only
    ):
        for name, term, tc in _whisper_terms():
            if term is None:
                rows.append(
                    {
                        "module": name,
                        "kind": "whisper",
                        "error": tc["export_error"],
                        "rule_fires": {},
                        "app_fires": {},
                    }
                )
                print(
                    f"[{name}] export FAILED: {tc['export_error']}",
                    flush=True,
                )
                continue
            term_cases[name] = tc
            wall = args.wall * 2
            row = _run_named(name, term, rules, wall)
            row["kind"] = "whisper"
            row["op_census"] = dict(_op_census(term))
            rows.append(row)
            print(
                f"[{name}] {row['seconds']}s iters={row['iterations']} "
                f"stop={row['stop']} enodes={row['n_enodes']} "
                f"fires={len(row['rule_fires'])}",
                flush=True,
            )

    # -- fragments ---------------------------------------------------
    if not args.no_fragments and (not args.only or "frag" in args.only):
        for name, mod, feed in _fragment_cases():
            if not keep(name):
                continue
            try:
                ir, leaves = TorchSource().to_ir(mod, feed)
            except Exception as exc:
                rows.append(
                    {
                        "module": name,
                        "kind": "frag",
                        "error": f"export: {type(exc).__name__}: {exc}",
                        "rule_fires": {},
                        "app_fires": {},
                    }
                )
                continue
            feeds = feed if isinstance(feed, tuple) else (feed,)
            tc = _term_case(name, ir, dict(leaves), tuple(feeds))
            term_cases[name] = tc
            row = _run_named(name, tc.term, rules, args.wall)
            row["kind"] = "frag"
            row["op_census"] = dict(_op_census(tc.term))
            rows.append(row)
            print(
                f"[{name}] {row['seconds']}s iters={row['iterations']} "
                f"stop={row['stop']} enodes={row['n_enodes']} "
                f"fires={len(row['rule_fires'])}",
                flush=True,
            )

    # -- mutants -----------------------------------------------------
    if (
        not args.no_mutants
        and args.mutants > 0
        and (
            not args.only or "mutant" in args.only or "zoo" in args.only
        )
    ):
        for base_name in _MUTANT_BASES:
            if not keep(f"mutant:{base_name}") and args.only:
                continue
            try:
                from mlopt_discovery import workload_gen as _wg
                from mlopt_discovery.impact import _iter_subterms

                if f"zoo:{base_name}" in term_cases:
                    base_tc = term_cases[f"zoo:{base_name}"]
                else:
                    _t, base_tc = _zoo_term(base_name)
                    term_cases[f"zoo:{base_name}"] = base_tc
                corpus = list(term_cases.values())
                # The corpus includes the whisper exports — a
                # shared-DAG model term whose unbounded ``shape_key``
                # tree-render OOMs (the same wall
                # ``model_corpus._probe_knobs`` bounds with
                # ``key_cap``).  Cap expansion for the novelty keys
                # and size the mutant node budget the way
                # ``model_corpus`` does (model nodes + headroom,
                # floored at _MAX_NODES).
                node_cap = max(
                    _wg._MAX_NODES,
                    min(
                        240,
                        len(_iter_subterms(base_tc.term))
                        + _MUTANT_HEADROOM,
                    ),
                )
                mutants, mstats = generate_mutants(
                    base_tc,
                    corpus,
                    args.mutants,
                    seed=args.mutant_seed,
                    max_nodes=node_cap,
                    key_cap=_MUTANT_KEY_CAP,
                )
                print(
                    f"[mutants:{base_name}] "
                    f"{mstats['accepted']}/{mstats['attempted']} "
                    f"accepted rejections={mstats['rejected']}",
                    flush=True,
                )
                for i, mtc in enumerate(mutants):
                    name = f"mutant:{base_name}:{i}"
                    if not keep(name):
                        continue
                    row = _run_named(name, mtc.term, rules, args.wall)
                    row["kind"] = "mutant"
                    row["op_census"] = dict(_op_census(mtc.term))
                    rows.append(row)
                    print(
                        f"[{name}] {row['seconds']}s "
                        f"iters={row['iterations']} "
                        f"stop={row['stop']} enodes={row['n_enodes']} "
                        f"fires={len(row['rule_fires'])}",
                        flush=True,
                    )
            except Exception as exc:
                rows.append(
                    {
                        "module": f"mutants:{base_name}",
                        "kind": "mutant",
                        "error": f"{type(exc).__name__}: {exc}",
                        "rule_fires": {},
                        "app_fires": {},
                    }
                )
                print(
                    f"[mutants:{base_name}] FAILED: {exc}", flush=True
                )

    # -- analysis ----------------------------------------------------
    report = analyze(rows, rules)
    report["config"] = {
        "rules": len(rules),
        "symmetry_budget": SYMMETRY_BUDGET,
        "max_nodes": args.max_nodes,
        "wall_s": args.wall,
        "only": args.only,
    }
    report["rows"] = rows

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[wrote] {out}", flush=True)

    # console digest
    ps = report["prim_summary"]
    print(
        f"[digest] prim folds: {ps['n_prim_rules']} in bank, "
        f"{ps['fired_anywhere']} fired somewhere, "
        f"{len(ps['never_fired'])} never fired",
        flush=True,
    )
    print(
        f"[digest] never-fired rules total: {len(report['never_fired'])} "
        f"({len(report['dead_machine'])} machine)",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
