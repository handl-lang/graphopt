"""Decoder-layer IR census vs law-LHS coverage probe.

Exports ONE whisper-base decoder layer (``decoder.layers[0]``, the
hook-captured decode-step inputs: seq len 1, eager attention, KV-cache
``EncoderDecoderCache``) to IR, then measures:

1. **Op census** — every op in the exported term, with counts.
2. **Law-LHS census** — the op multiset each shipped rule's ``lhs``
   pattern mentions (``default_rules()`` = core DEFAULT + carriers).
3. **Match-level coverage** — for every rule, the number of
   substitutions its LHS produces over the saturated-to-fixed-point
   e-graph, and how many survive ``check``/``derive``.  Separates
   "no LHS coverage" from "matched but vetoed".

Usage::

    .venv/bin/python bench/drivers/probes/decoder_layer_census.py
    .venv/bin/python bench/drivers/probes/decoder_layer_census.py search
"""

from __future__ import annotations

import copy
import os
import resource
import sys
import time
from collections import Counter

resource.setrlimit(resource.RLIMIT_AS, (9 << 30, 9 << 30))

import torch
import torch.nn as nn

MODEL = "openai/whisper-base"


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


class _LayerWrap(nn.Module):
    """Replay the hook-captured layer-0 decode inputs.

    ``hs`` is the traced input (embed+pos output, (1,1,512)); the causal
    mask and encoder hidden states ride as buffers (constants — fine
    for a census).  A fresh ``EncoderDecoderCache`` is built per call so
    the KV-cache ``cat`` plumbing is part of the traced graph.
    """

    def __init__(self, layer, cfg, mask, ehs):
        super().__init__()
        self.layer = layer
        self.cfg = cfg
        self.register_buffer("mask", mask)
        self.register_buffer("ehs", ehs)

    def forward(self, hs):
        from transformers.cache_utils import (
            DynamicCache,
            EncoderDecoderCache,
        )

        c = copy.deepcopy(self.cfg)
        c.num_hidden_layers = self.cfg.decoder_layers
        cache = EncoderDecoderCache(
            DynamicCache(config=c), DynamicCache(config=c)
        )
        return self.layer(
            hs,
            self.mask,
            self.ehs,
            None,
            past_key_values=cache,
            use_cache=True,
        )


def capture_inputs(model):
    """Run one decode step; return (hs, mask, ehs) seen by layer 0."""
    x = torch.randn(1, 80, 3000)
    with torch.no_grad():
        ehs = model.model.encoder(x).last_hidden_state
        start = torch.tensor(
            [[model.generation_config.decoder_start_token_id or 50257]]
        )
    cap = {}

    def pre(mod, args, kwargs):
        cap["args"] = args
        cap["kwargs"] = kwargs

    h = model.model.decoder.layers[0].register_forward_pre_hook(
        pre, with_kwargs=True
    )
    try:
        with torch.no_grad():
            model.model.decoder(
                input_ids=start, encoder_hidden_states=ehs
            )
    finally:
        h.remove()
    hs, mask = cap["args"][0], cap["args"][1]
    ehs_in = cap["args"][2]
    return hs, mask, ehs_in


def op_census(term, counts=None):
    """Count ops over the (DAG-shared) term — dedup by object id."""
    from mlopt_core.ir import Op

    if counts is None:
        counts = Counter()
    seen = set()

    def walk(t):
        if not isinstance(t, Op) or id(t) in seen:
            return
        seen.add(id(t))
        counts[t.op] += 1
        for a in t.args:
            walk(a)

    walk(term)
    return counts, len(seen)


def pattern_ops(pat, counts=None):
    """Count ops in a rule-LHS/RHS *pattern* (strings are metavars)."""
    from mlopt_core.ir import Op

    if counts is None:
        counts = Counter()
    if isinstance(pat, Op):
        counts[pat.op] += 1
        for a in pat.args:
            pattern_ops(a, counts)
    return counts


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "census"
    import mlopt_carriers  # noqa: F401 — registers the CARRIERS preset
    import mlopt_torch  # noqa: F401 — torch bindings for export
    from mlopt_core.egraph import EGraph
    from mlopt_orchestrator.optimize import default_rules
    from mlopt_torch.adapters import TorchSource
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    hs, mask, ehs = capture_inputs(model)
    print(
        f"[inputs] hs={tuple(hs.shape)} mask={tuple(mask.shape)} "
        f"ehs={tuple(ehs.shape)} rss={rss_gb():.1f}GB",
        flush=True,
    )
    wrap = _LayerWrap(model.model.decoder.layers[0], model.config, mask, ehs)

    t0 = time.time()
    ir, leaves = TorchSource().to_ir(wrap, (hs,))
    print(
        f"[export] {time.time() - t0:.1f}s root={ir.root.op} "
        f"leaves={len(leaves)} rss={rss_gb():.1f}GB",
        flush=True,
    )

    counts, nnodes = op_census(ir.root)
    print(f"[ir] distinct ops={nnodes} unique-op-kinds={len(counts)}")
    for op, n in counts.most_common():
        print(f"    {op:24s} {n}")

    rules = list(default_rules())
    print(f"[rules] {len(rules)}")
    extra = os.environ.get("EXTRA_RULES", "")
    if extra:
        from mlopt_core.laws.priorart import PRIOR_ART_RULES
        from mlopt_core.laws.attention import ATTENTION_RULES

        pool = {r.name: r for r in PRIOR_ART_RULES + list(ATTENTION_RULES)}
        add = [pool[n] for n in extra.split(",") if n in pool]
        missing = [n for n in extra.split(",") if n not in pool]
        if missing:
            print(f"[extra] MISSING: {missing}")
        rules += add
        print(f"[extra] +{len(add)} -> {len(rules)} total")

    # ---- law LHS op census -------------------------------------
    law_ops = Counter()
    for r in rules:
        law_ops.update(pattern_ops(r.lhs))
    print("[law-lhs ops]", dict(law_ops.most_common()))

    # ops present in IR but never on any LHS
    uncovered = {op: n for op, n in counts.items() if op not in law_ops}
    print("[ir ops with NO lhs coverage]:", uncovered)

    # ---- match-level coverage ----------------------------------
    eg = EGraph()
    root = eg.add_term(ir.root)
    print(
        f"[eg] enodes={eg.n_enodes} classes={eg.n_classes} "
        f"rss={rss_gb():.1f}GB",
        flush=True,
    )

    # Per-rule: count LHS substitutions over all classes, then how
    # many survive check/derive (replay apply_rule's guard logic).
    rows = []
    for r in rules:
        prog = eg._prog_for(r.lhs)
        cands = eg._candidate_classes(prog, None)
        nmatch = 0
        nok = 0
        for eid in cands:
            for subst in eg._iter_matches(prog, eid, None):
                nmatch += 1
                if r.check is None and r.derive is None:
                    nok += 1
                    continue
                bound = {
                    k: (
                        v
                        if k.startswith("$attr:")
                        else eg._any_term_cached(v)
                    )
                    for k, v in subst.items()
                }
                if any(
                    v is None
                    for k, v in bound.items()
                    if not k.startswith("$attr:")
                ):
                    continue
                if r.check is not None and not r.check(bound):
                    continue
                if r.derive is not None and r.derive(bound) is None:
                    continue
                nok += 1
        if nmatch or nok:
            rows.append((r.name, nmatch, nok))
    rows.sort(key=lambda t: -t[1])
    print("[rule matches] name  lhs-substs  survive-check")
    for name, nm, nok in rows:
        print(f"    {name:44s} {nm:6d} {nok:6d}")

    if mode == "search":
        t0 = time.time()
        stats = eg.run(rules, root, max_iterations=3)
        print(
            f"[search] {time.time() - t0:.1f}s enodes={eg.n_enodes} "
            f"classes={eg.n_classes} stats={stats}",
            flush=True,
        )


if __name__ == "__main__":
    main()
