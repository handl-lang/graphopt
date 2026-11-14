"""Per-rule timing inside the REAL ``optz.search`` decoder path.

Wraps ``EGraph.apply_rule`` so every rule application logs wall
time + enode/class deltas while ``Optimizer.search`` runs on the
whisper-base decoder step (CPU fp32).  ``faulthandler`` dumps stacks
every 120 s as a backstop.  ``RLIMIT_AS`` caps the probe at 9 GiB.

Usage::

    .venv/bin/python bench/drivers/probes/decoder_search_rules.py
"""

from __future__ import annotations

import resource
import sys
import time

resource.setrlimit(resource.RLIMIT_AS, (9 << 30, 9 << 30))

import torch
import torch.nn as nn

MODEL = "openai/whisper-base"
LOG = "/tmp/dec_rules.log"


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


class _DecWrap(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, ids, ehs):
        return self.dec(
            input_ids=ids, encoder_hidden_states=ehs
        ).last_hidden_state


def main() -> None:
    import faulthandler

    faulthandler.dump_traceback_later(120, repeat=True)
    from mlopt_core.egraph.core import EGraph as _EG
    from transformers import WhisperForConditionalGeneration
    from mlopt_orchestrator.optimize import Optimizer
    from mlopt_torch.backend import TorchBackend

    orig_apply = _EG.apply_rule
    orig_run = _EG.run
    state = {"iter": -1, "rule": None}

    def timed_run(self, rules, root_eid, **kw):
        state["iter"] += 1
        t0 = time.time()
        log(f"[run {state['iter']}] start enodes={self.n_enodes} "
            f"kw={ {k: v for k, v in kw.items() if k in ('max_iterations','max_nodes','stop')} }")
        out = orig_run(self, rules, root_eid, **kw)
        log(f"[run {state['iter']}] done {time.time() - t0:.1f}s "
            f"enodes={self.n_enodes} stats={out}")
        return out

    SKIP = set()
    if len(sys.argv) > 1:
        SKIP = set(sys.argv[1].split(","))

    def timed_apply(self, rule, root_eid, search=None, enode_budget=None):
        n0, c0 = self.n_enodes, self.n_classes
        t0 = time.time()
        if rule.name in SKIP:
            log(f"  ?? {rule.name} SKIPPED")
            return False
        log(f"  .. {rule.name} START enodes={self.n_enodes} "
            f"search={'full' if search is None else len(search)} "
            f"budget={enode_budget}")
        try:
            _wrap_rule_hooks(rule)
            out = orig_apply(
                self, rule, root_eid, search=search,
                enode_budget=enode_budget,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  RULE {rule.name} RAISED {exc!r} "
                f"after {time.time() - t0:.1f}s")
            raise
        dt = time.time() - t0
        de, dc = self.n_enodes - n0, self.n_classes - c0
        if dt > 0.2 or de or dc:
            log(f"  RULE {rule.name:40s} {dt:7.2f}s "
                f"+{de:6d}e +{dc:6d}c "
                f"search={'full' if search is None else len(search)} "
                f"budget={enode_budget}")
        return out

    # inner-layer instrumentation: which sub-call eats the time
    orig_iter = _EG._iter_matches
    orig_inst = _EG._instantiate
    orig_min = _EG._min_term

    def timed_iter(self, prog, eid, limit):
        t0 = time.time()
        n = 0
        for b in orig_iter(self, prog, eid, limit):
            n += 1
            yield b
        dt = time.time() - t0
        if dt > 0.5 or n > 1000:
            log(f"    _iter_matches eid={eid} yielded={n} {dt:.2f}s")

    def timed_inst(self, pn, subst):
        t0 = time.time()
        out = orig_inst(self, pn, subst)
        dt = time.time() - t0
        if dt > 0.5:
            log(f"    _instantiate {dt:.2f}s")
        return out

    def timed_min(self, eid, memo, _seen=frozenset()):
        t0 = time.time()
        out = orig_min(self, eid, memo, _seen)
        dt = time.time() - t0
        if dt > 0.5:
            sz = out[1] if out else "?"
            log(f"    _min_term eid={eid} {dt:.2f}s size={sz}")
        return out

    orig_union = _EG.union

    def timed_union(self, a, b, **kw):
        t0 = time.time()
        out = orig_union(self, a, b, **kw)
        dt = time.time() - t0
        if dt > 0.5:
            log(f"    union({a},{b}) {dt:.2f}s")
        return out

    _EG.union = timed_union

    # wrap every rule's check/derive with timers (once, lazily)
    wrapped_rules = set()

    def _wrap_rule_hooks(rule):
        if rule.name in wrapped_rules:
            return
        wrapped_rules.add(rule.name)
        ck, dv = rule.check, rule.derive
        if ck is not None:
            def _ck(b, _f=ck, _n=rule.name):
                t0 = time.time()
                r = _f(b)
                dt = time.time() - t0
                if dt > 0.2:
                    log(f"    check {_n} {dt:.2f}s -> {r}")
                return r
            try:
                object.__setattr__(rule, "check", _ck)
            except Exception:
                pass
        if dv is not None:
            def _dv(b, _f=dv, _n=rule.name):
                t0 = time.time()
                r = _f(b)
                dt = time.time() - t0
                if dt > 0.2:
                    log(f"    derive {_n} {dt:.2f}s")
                return r
            try:
                object.__setattr__(rule, "derive", _dv)
            except Exception:
                pass

    _EG._iter_matches = timed_iter
    _EG._instantiate = timed_inst
    _EG._min_term = timed_min
    _EG.run = timed_run
    _EG.apply_rule = timed_apply

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    x = torch.randn(1, 80, 3000)
    with torch.no_grad():
        ehs = model.model.encoder(x).last_hidden_state
        start = torch.tensor(
            [[model.generation_config.decoder_start_token_id or 50257]]
        )
    optz = Optimizer(backend=TorchBackend())
    t0 = time.time()
    try:
        res = optz.search(_DecWrap(model.model.decoder), (start, ehs),
                          max_iterations=3)
        log(f"search done {time.time() - t0:.1f}s")
        log(str({k: v for k, v in res.stats.items()
                 if k != "param_sharing"}))
    except Exception as exc:  # noqa: BLE001
        log(f"search raised {exc!r} after {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
