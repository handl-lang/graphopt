"""Phase-timed decoder search probe.

Wraps each phase of ``mlopt_orchestrator.optimize.search`` with a
timer + RSS report, then runs the whisper-base decoder step on CPU.
Complements ``decoder_search_probe.py`` (which showed plain
saturation fixed-points in <1 s): the >20 min wall reported in
``docs/reports/whisper_experiment.md`` must live in the non-local
passes or extraction, not ``EGraph.run``.

Usage::

    .venv/bin/python bench/drivers/probes/decoder_search_phases.py [seconds]
"""

from __future__ import annotations

import resource
import sys
import time

import torch
import torch.nn as nn

MODEL = "openai/whisper-base"


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def wrap(mod, name):
    fn = getattr(mod, name)

    def inner(*a, **kw):
        t0 = time.time()
        print(f"  >> {name} start rss={rss_gb():.1f}GB", flush=True)
        try:
            out = fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  << {name} RAISED {exc!r} "
                f"after {time.time() - t0:.1f}s",
                flush=True,
            )
            raise
        print(
            f"  << {name} done {time.time() - t0:.1f}s "
            f"rss={rss_gb():.1f}GB",
            flush=True,
        )
        return out

    setattr(mod, name, inner)
    return fn


class _DecWrap(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, ids, ehs):
        return self.dec(
            input_ids=ids, encoder_hidden_states=ehs
        ).last_hidden_state


def main() -> None:
    import mlopt_orchestrator.optimize as opt
    from mlopt_core.egraph.core import EGraph
    from mlopt_core.laws import pairing
    from mlopt_torch.adapters import TorchSource

    # Saturation engine + non-local passes + extraction.  The pairing
    # helpers are ``from``-imported into ``optimize``'s namespace, so
    # wrap them THERE (and in pairing for direct callers).
    import faulthandler

    faulthandler.dump_traceback_later(90, repeat=True)
    wrap(EGraph, "run")
    for mod in (opt, pairing):
        for name in (
            "pair_shared_input_linears",
            "pair_shared_input_convs",
            "share_duplicate_param_slices",
            "share_duplicate_params",
        ):
            if hasattr(mod, name):
                wrap(mod, name)
    wrap(opt, "_carrier_lifts")
    wrap(opt, "_select_best_term")
    # export_to_ir lives behind TorchSource.to_ir
    wrap(TorchSource, "to_ir")

    from transformers import WhisperForConditionalGeneration
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    x = torch.randn(1, 80, 3000)
    with torch.no_grad():
        ehs = model.model.encoder(x).last_hidden_state
        start = torch.tensor(
            [[model.generation_config.decoder_start_token_id or 50257]]
        )
    print(f"[setup] rss={rss_gb():.1f}GB", flush=True)
    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    try:
        res = optz.search(
            _DecWrap(model.model.decoder), (start, ehs),
            max_iterations=3,
        )
        print(f"search done {time.time() - t0:.1f}s", flush=True)
        print({k: v for k, v in res.stats.items()
               if k != "param_sharing"}, flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"search raised {exc!r} after {time.time() - t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
