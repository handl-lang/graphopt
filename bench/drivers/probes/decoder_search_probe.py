"""Decoder-scale e-graph explosion probe — measurement driver.

Reproduces the whisper_experiment.md decoder wall with instrumentation:
exports ``whisper-base``'s decoder step (stateless, no KV cache) to IR
on **CPU** (the e-graph is host-side anyway), then saturates with the
pipeline's composed ``DEFAULT_RULES`` while logging, per rule per
iteration: wall time, enodes added, e-classes, dirty-frontier size.

Usage::

    .venv/bin/python bench/drivers/probes/decoder_search_probe.py [iters]
    .venv/bin/python bench/drivers/probes/decoder_search_probe.py export-only

Scratch-quality, like ``whisper_probe.py``.  Findings land in
``docs/reports/decoder_search_scaling.md``.
"""

from __future__ import annotations

import resource
import sys
import time

import torch
import torch.nn as nn

MODEL = "openai/whisper-base"
N_MELS = 80
N_FRAMES = 3000  # whisper requires exactly 30 s of log-mel
MEM_CAP_MB = 8 * 1024  # stay under the 11 GB host


class _DecWrap(nn.Module):
    """Decoder step -> logits only (see whisper_probe.py)."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, ids, ehs):
        return self.dec(
            input_ids=ids, encoder_hidden_states=ehs
        ).last_hidden_state


def load_cpu():
    from transformers import WhisperForConditionalGeneration

    return WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()


def export_ir():
    """Export the decoder step; return (ir, leaves, root_eid-ready eg)."""
    from mlopt_core.egraph import EGraph
    from mlopt_orchestrator.optimize import default_rules
    from mlopt_torch.adapters import TorchSource

    model = load_cpu()
    x = torch.randn(1, N_MELS, N_FRAMES)
    with torch.no_grad():
        ehs = model.model.encoder(x).last_hidden_state
        start = torch.tensor(
            [[model.generation_config.decoder_start_token_id or 50257]]
        )
    mod = _DecWrap(model.model.decoder)
    t0 = time.time()
    ir, leaves = TorchSource().to_ir(mod, (start, ehs))
    print(f"[export] {time.time() - t0:.1f}s", flush=True)
    rules = default_rules()
    print(f"[rules] {len(rules)} rules", flush=True)
    eg = EGraph()
    root = eg.add_term(ir.root)
    print(
        f"[ir] root enode={root} leaves={len(leaves)}",
        flush=True,
    )
    return eg, rules, root, leaves


def profile(iters: int = 4, max_nodes: int = 400_000) -> None:
    from mlopt_core.egraph.core import EGraph as _EG
    from mlopt_core.laws import tags

    eg, rules, root, leaves = export_ir()
    orig_apply = _EG.apply_rule
    log = []

    def timed_apply(self, rule, root_eid, search=None, enode_budget=None):
        n0, c0 = self.n_enodes, self.n_classes
        t0 = time.time()
        try:
            out = orig_apply(
                self, rule, root_eid, search=search,
                enode_budget=enode_budget,
            )
        except Exception as exc:  # noqa: BLE001 - probe records, reraises
            log.append(
                (rule.name, time.time() - t0, self.n_enodes - n0,
                 self.n_classes - c0, f"ERR {exc!r}")
            )
            raise
        log.append(
            (rule.name, time.time() - t0, self.n_enodes - n0,
             self.n_classes - c0, "")
        )
        return out

    _EG.apply_rule = timed_apply
    try:
        # One iteration per run() call: stats + per-rule table stay
        # attributable; _applied_rules/_dirty persist across calls.
        for i in range(iters):
            n0 = eg.n_enodes
            t0 = time.time()
            stats = eg.run(
                rules, root, max_iterations=1, max_nodes=max_nodes,
                rule_budgets={
                    r.name: 2048
                    for r in rules.tagged(tags.EXPANSIVE)
                },
            )
            dt = time.time() - t0
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(
                f"[iter {i}] {dt:.1f}s enodes {n0}->{eg.n_enodes} "
                f"classes={eg.n_classes} rss={rss:.1f}GB "
                f"stop={stats['stop']}",
                flush=True,
            )
            # per-rule table for this iteration
            for name, dt_r, de, dc, err in log:
                if de or dt_r > 0.5 or err:
                    print(
                        f"    {name:38s} {dt_r:8.2f}s "
                        f"+{de:7d} enodes +{dc:7d} cls {err}",
                        flush=True,
                    )
            log.clear()
            if stats["stop"] != "max_iterations" and i > 0:
                break
    finally:
        _EG.apply_rule = orig_apply


def main() -> None:
    resource.setrlimit(
        resource.RLIMIT_AS, (MEM_CAP_MB * 1024 * 1024,) * 2
    )
    arg = sys.argv[1] if len(sys.argv) > 1 else "4"
    if arg == "export-only":
        eg, rules, root, leaves = export_ir()
        return
    iters = int(arg)
    profile(iters=iters)


if __name__ == "__main__":
    main()
