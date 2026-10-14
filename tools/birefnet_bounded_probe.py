"""BiRefNet bounded-rewrite probe — the ``error_budget=`` lane on a real pretrained vision model.

Context: ``bench/suites/bounded/bounded_e2e.py`` delivered
certified-bounded members on stories15M (1.41x vs eager at budget
1e-2, every budget inside the fp32 equivalence gate).  The BiRefNet
trial (commit ``34607e0``) exported the real 215M checkpoint but the
*exact* lane was FLOP-neutral there — attention already calls
``F.sdpa``, the decoder is conv traffic.  This probe opens the
**bounded** lane on the real pretrained model: per budget cell it
runs the compositional pipeline with ``detect_specials=True`` and
``error_budget=B`` (arming the certified ``elide_bounded`` /
``zero_bounded`` weight specials — 2-D ``linear``/``matmul``
projections only), verifies, and times the delivered module against
eager and ``torch.compile``.

Model: the REAL pretrained checkpoint — ``ZhengPeng7/BiRefNet`` on
HF (Swin-v1-L backbone + ASPPDeformable decoder, 220M params, fp32
safetensors ~886MB, loaded through ``trust_remote_code`` with the
cached repo snapshot).  Scratch deps beyond the bench group —
installed ``--no-deps`` into the worktree venv::

    uv pip install --no-deps timm einops kornia kornia-rs safetensors
    uv pip install transformers torchvision

(kornia/timm/torchvision are imported at module level by the HF
``birefnet.py`` remote code; ``laplacian`` itself is only reached in
the training branch — eval never calls it).

Block selection: BiRefNet's ``mul_scl_ipt='cat'`` calls
``model.bb`` TWICE per forward — once on ``x``, once on
``x/2`` — so any grafted container that bakes its captured
resolution (a ``BasicLayer`` stage, a ``SwinTransformerBlock``:
reshape/window constants, plus ``self.H``/``self.W`` mutated by the
parent per call) silently breaks the second call and the
recomposed model is unusable.  ``_block_pred`` therefore declines
``BasicLayer`` / ``SwinTransformerBlock`` / ``BasicDecBlk`` /
``_ASPPModuleDeformable`` (the last two also contain
``deform_conv2d``, which has no torch lowering binding) so
selection descends to the *shape-generic* leaves — where the
bounded specials live anyway: every ``nn.Linear`` leaf exports as
exactly the ``linear(data, W[, b])`` projection site
``offer_weight_specials`` scans.

Per budget (``--budgets``, default ``none,1e-4,1e-3,3e-3,1e-2``):

* optimize on CPU (the bounded-elide row loop is ~50x slower per
  launch on CUDA — weight-space bounds are device-invariant; the
  delivered module is re-verified on ``--device``);
* ledger: bounded offers, delivered members, certified bound
  totals, honest bound-gate declines;
* fresh-input verify + held-out proxy (seeded random + checkerboard
  images): rel/max-abs diff and thresholded-mask IoU;
* timing on ``--device``: source eager / source compile /
  delivered eager / delivered compile, with steady-state peak CUDA
  memory per arm.

Structural census (the pending question): a forward hook captures
``model.decoder``'s real input, the decoder is exported on its own,
and the IR op histogram is printed — settling whether the decoder
carries ``batch_norm`` nodes (conv+BN fusion targets) or is
LN/conv-only.  A ``--full-ir-census`` arm does the same for the
whole-model export.

Usage::

    .venv/bin/python tools/birefnet_bounded_probe.py --device cuda
    .venv/bin/python tools/birefnet_bounded_probe.py --budgets none,1e-3
"""
# ruff: noqa: RUF003 -- ×, −, ≈ in docstrings/comments are
# deliberate math notation (bounded_e2e.py convention).

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from catopt_discovery.play import _time_module
from catopt_orchestrator import Compositional, Optimizer
from catopt_torch.adapters import TorchSource
from catopt_torch.backend import TorchBackend
from catopt_torch.composer import TorchComposer
from catopt_torch.report import verify_equiv

try:
    from transformers import AutoModelForImageSegmentation
except ImportError:  # environment guard — see module docstring
    AutoModelForImageSegmentation = None

_BUDGET_DEFAULT = "none,1e-4,1e-3,3e-3,1e-2"
_HF_REPO = "ZhengPeng7/BiRefNet"


class _CompileTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _CompileTimeout()


# ---------------------------------------------------------------------------
#  Model loading + held-out probe images
# ---------------------------------------------------------------------------


def _load_birefnet(dtype: torch.dtype = torch.float32) -> nn.Module:
    """Load the real pretrained BiRefNet from the HF cache."""
    if AutoModelForImageSegmentation is None:
        raise RuntimeError(
            "transformers not installed — see module docstring for "
            "the scratch-dep install lines"
        )
    model = AutoModelForImageSegmentation.from_pretrained(
        _HF_REPO, trust_remote_code=True, dtype=dtype
    )
    return model.eval()


def _heldout_images(res: int, n: int, device: torch.device):
    """Seeded random images + one checkerboard (structured edge case).

    No eval dataset ships with the checkpoint; the proxy is
    distributional, matching bounded_e2e's random+periodic prompt
    convention — random tensors exercise the diffuse regime, the
    checkerboard drives the high-contrast edge regime where a mask
    boundary either survives or doesn't.
    """
    g = torch.Generator().manual_seed(20261014)
    imgs = [torch.rand(1, 3, res, res, generator=g) for _ in range(n)]
    yy, xx = torch.meshgrid(
        torch.arange(res), torch.arange(res), indexing="ij"
    )
    cb = ((xx // 24 + yy // 24) % 2).float()
    imgs.append(cb.expand(1, 3, res, res).contiguous())
    return [im.to(device) for im in imgs]


def _mask_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """IoU of the thresholded (sigmoid>0) masks — the task proxy."""
    ma = a.float().sigmoid() > 0.5
    mb = b.float().sigmoid() > 0.5
    inter = (ma & mb).sum().item()
    union = (ma | mb).sum().item()
    return inter / union if union else 1.0


def _proxy_eval(ref_mod, opt_mod, images) -> dict:
    """Mask agreement of ``opt_mod`` vs ``ref_mod`` on held-out images."""
    rels, abses, ious = [], [], []
    with torch.no_grad():
        for im in images:
            r = ref_mod(im)
            o = opt_mod(im)
            r = r[0] if isinstance(r, (list, tuple)) else r
            o = o[0] if isinstance(o, (list, tuple)) else o
            r, o = r.float(), o.float()
            d = (r - o).abs()
            abses.append(d.max().item())
            rels.append((d / r.abs().clamp_min(1e-6)).max().item())
            ious.append(_mask_iou(r, o))
    return {
        "max_abs": max(abses),
        "max_rel": max(rels),
        "min_iou": min(ious),
        "n_images": len(images),
    }


# ---------------------------------------------------------------------------
#  Op census — the decoder BN question
# ---------------------------------------------------------------------------


def _op_hist(term) -> Counter:
    """Op-name histogram of an IR term (args-first walk).

    The term is a DAG — shared subterms are visited once (interned
    node identity), so the counts are the *spelled* graph's, not a
    tree expansion's.
    """
    from catopt_core.ir import Op

    acc: Counter = Counter()
    seen: set = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if isinstance(t, Op):
            acc[t.op] += 1
            stack.extend(t.args)
    return acc


def _decoder_census(model, x) -> dict:
    """Export ``model.decoder`` on its real captured input; histogram.

    A forward hook grabs the ``features`` list the decoder is
    actually called with, then the decoder alone goes through
    ``TorchSource.to_ir`` — the answer to "does the decoder contain
    batch_norm nodes or is it LN/conv-only" in the model's own
    op vocabulary.
    """
    captured = {}

    def hook(mod, args, kwargs, out):
        captured["args"] = args

    h = model.decoder.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        model(x)
    h.remove()
    args = captured.get("args", ())
    ir = None
    last_err = None
    for candidate in (args, args[0] if len(args) == 1 else args):
        try:
            ir, _leaves = TorchSource().to_ir(model.decoder, candidate)
            break
        except Exception as e:  # export-spelling probe — keep trying
            last_err = e
    if ir is None:
        return {"error": f"{type(last_err).__name__}: {last_err}"}
    hist = _op_hist(ir.root)
    # Where the BN modules sit in the tree (module-side corroboration)
    bn_mods = [
        n
        for n, mod in model.named_modules()
        if isinstance(mod, nn.BatchNorm2d)
    ]
    return {
        "n_enodes": sum(hist.values()),
        "ops": dict(hist.most_common()),
        "batch_norm_nodes": hist.get("batch_norm", 0),
        "bn_modules": bn_mods,
    }


def _full_census(model, x) -> dict:
    """Whole-model export op histogram (best-effort)."""
    ir, _leaves = TorchSource().to_ir(model, x)
    hist = _op_hist(ir.root)
    return {
        "n_enodes": sum(hist.values()),
        "ops": dict(hist.most_common()),
        "batch_norm_nodes": hist.get("batch_norm", 0),
    }


# ---------------------------------------------------------------------------
#  Block selection — descend into the BasicLayer stage wrappers
# ---------------------------------------------------------------------------


#: Containers declined by :func:`_block_pred` so block selection
#: descends to their shape-generic leaves instead.
#:
#: * ``BasicLayer`` / ``SwinTransformerBlock`` — a delivered block is
#:   *shape-specialized for its first captured call*, but
#:   ``mul_scl_ipt='cat'`` calls ``model.bb`` twice per forward — once
#:   on ``x``, once on ``x/2`` — so any grafted stage/block silently
#:   breaks the second (half-resolution) call: the e2e check records
#:   the error and the recomposed model is unusable.  (Per-block
#:   calls also read ``self.H``/``self.W`` mutated by the parent —
#:   the captured replay sees the stale half-res values and the
#:   block's own ``assert L == H*W`` fires.)
#: * ``BasicDecBlk`` / ``_ASPPModuleDeformable`` — the exported graph
#:   contains ``deform_conv2d``, which has no torch lowering binding.
#:
#: Their *leaves* — the Linears the bounded specials actually scan,
#: plus shape-generic conv/BN/LN/ReLU — are always selected by the
#: composer, so the bounded lane's offer space is unaffected.
_NO_GRAFT = frozenset(
    {
        "BasicLayer",
        "SwinTransformerBlock",
        "BasicDecBlk",
        "_ASPPModuleDeformable",
    }
)


def _block_pred(parent: nn.Module, name: str, mod: nn.Module) -> bool:
    """ModuleList/Sequential children, minus the no-graft classes."""
    if isinstance(parent, (nn.ModuleList, nn.Sequential)):
        return type(mod).__name__ not in _NO_GRAFT
    return False


#: The bounded lane's offer space — ``offer_weight_specials`` scans
#: 2-D ``linear``/``matmul`` projection sites only ("non-2-D weights
#: out of scope"), so conv/BN/LN leaf searches can never produce a
#: bounded member.  Restricting the composer's block selection to
#: ``nn.Linear`` leaves is what makes the run *survivable* here:
#: ``capture_inputs`` keeps a detached clone of every selected
#: block's input AND a live reference to its first output, which on
#: a 220M-param model at 384² costs ~6 GB host RAM with the full
#: ~500-block selection (OOM-killed, measured) — versus ~0.5 GB for
#: the ~100 Linear leaves.
_LEAF_CLASSES: tuple = (nn.Linear,)


class _LeafComposer(TorchComposer):
    """``blocks()`` filtered to the bounded lane's leaf classes.

    The predicate still decides where the walk descends
    (:func:`_block_pred` above); this filter only narrows which of
    the selected blocks are *searched and grafted*.
    """

    def blocks(self, model, *, predicate=None):
        """Return selected blocks that are ``_LEAF_CLASSES``."""
        return [
            (n, b)
            for n, b in super().blocks(model, predicate=predicate)
            if isinstance(b, _LEAF_CLASSES)
        ]


# ---------------------------------------------------------------------------
#  Bounded ledger + one budget cell (bounded_e2e conventions)
# ---------------------------------------------------------------------------


def _bounded_ledger(stats: dict) -> dict:
    """Roll the per-block bound records into one cell ledger."""
    per_block: dict[str, dict] = {}
    n_offers = 0
    n_bounded_offers = 0
    for name, rep in (stats.get("blocks") or {}).items():
        st = rep.get("stats") or {}
        specials = st.get("weight_specials") or []
        bounds = st.get("error_bounds") or []
        n_offers += len(specials)
        n_bounded_offers += sum(
            1 for s in specials if (s.get("error_bound") or 0.0) > 0
        )
        per_block[name] = {
            "status": rep.get("status"),
            "cache": rep.get("cache"),
            "rel_diff": rep.get("rel_diff"),
            "error": rep.get("error"),
            "time_s": round(rep.get("time_s") or 0.0, 1),
            "bound_total": st.get("error_bound_total"),
            "bounds": [
                {
                    k: e.get(k)
                    for k in (
                        "rule",
                        "bound",
                        "norm",
                        "output_bound",
                        "site_input_norm",
                        "measured_max_rel",
                    )
                }
                for e in bounds
            ],
            "specials_offered": [
                {
                    "param": s.get("param"),
                    "kind": s.get("kind"),
                    "error_bound": s.get("error_bound"),
                }
                for s in specials
            ],
            "bounds_honored": st.get("error_bounds_honored"),
        }
    return {
        "per_block": per_block,
        "n_special_offers": n_offers,
        "n_bounded_offers": n_bounded_offers,
    }


def _bench(mod, x, reps: int) -> dict:
    """Median latency + steady-state peak memory for *mod*.

    ``_time_module`` runs the module bare — grad must be OFF for the
    whole loop or the autograd graph of a 220M-param model
    triples the activation footprint (the OOM measured on the RTX
    2050 was this, not weights).
    """
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        for _ in range(3):
            mod(x)
        if x.is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        us = _time_module(mod, (x,), reps)
        peak = torch.cuda.max_memory_allocated() if x.is_cuda else 0
    finally:
        torch.set_grad_enabled(prev)
    return {"us": us, "peak_mb": peak / 2**20}


def _try_compile(mod, x, budget_s: float):
    """``torch.compile`` with a wall-clock timeout; (mod, status)."""
    if not hasattr(signal, "SIGALRM"):
        try:
            cm = torch.compile(mod)
            with torch.no_grad():
                cm(x)
            return cm, "compiled"
        except Exception as e:
            return None, f"compile failed: {type(e).__name__}: {e}"
    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, budget_s)
    t0 = time.perf_counter()
    try:
        cm = torch.compile(mod)
        with torch.no_grad():
            cm(x)
            cm(x)
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        return cm, f"compiled in {time.perf_counter() - t0:.1f}s"
    except _CompileTimeout:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        torch._dynamo.reset()
        return None, f"compile TIMEOUT >{budget_s:.0f}s"
    except Exception as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        torch._dynamo.reset()
        return None, f"compile failed: {type(e).__name__}: {e}"


def run_budget_cell(
    search_model,
    ref_model,
    x_search,
    x_dev,
    budget,
    label,
    args,
    dev,
    images,
) -> dict:
    """Optimize under ``budget``, verify, time, pack the record."""
    print(f"\n=== budget={label} ===", flush=True)
    rec: dict = {"name": f"budget={label}", "budget": label}

    t0 = time.time()
    opt_mod = None
    try:
        with torch.no_grad():
            opt_mod, stats = Optimizer(
                backend=TorchBackend(composer=_LeafComposer())
            ).optimize(
                search_model,
                x_search,
                strategy=Compositional(block_pred=_block_pred),
                detect_specials=True,
                error_budget=budget,
                max_iterations=int(args.max_iterations),
                max_enodes=int(args.max_enodes),
                verbose=False,
            )
        rec["opt_s"] = round(time.time() - t0, 1)
        rec["n_optimized"] = stats.get("n_optimized")
        rec["n_blocks"] = stats.get("n_blocks")
        rec["cache"] = stats.get("cache")
        rec["e2e"] = stats.get("end_to_end")
        rec.update(_bounded_ledger(stats))
        rec["n_bounds_delivered"] = sum(
            len(b["bounds"]) for b in rec["per_block"].values()
        )
        rec["bound_total_max"] = max(
            (
                b["bound_total"] or 0.0
                for b in rec["per_block"].values()
            ),
            default=0.0,
        )
        # Honest bound-gate declines drop their search stats — recover
        # the propagated bound from the driver's error string.
        declines = []
        for bn, b in rec["per_block"].items():
            err = b.get("error") or ""
            m = re.search(
                r"rel diff (\S+) \(accepted bound (\S+)\)", err
            )
            if m:
                declines.append(
                    {
                        "block": bn,
                        "measured_rel": float(m.group(1)),
                        "accepted_bound": float(m.group(2)),
                    }
                )
        if declines:
            rec["bound_declines"] = declines
        statuses = Counter(
            b["status"] for b in rec["per_block"].values()
        )
        rec["status_counts"] = dict(statuses)
        print(
            f"  optimize: {rec['opt_s']}s "
            f"opt={rec['n_optimized']}/{rec['n_blocks']} "
            f"cache={rec['cache']} "
            f"offers={rec['n_bounded_offers']} "
            f"delivered={rec['n_bounds_delivered']} "
            f"bound_max={rec['bound_total_max']:.3e} "
            f"statuses={dict(statuses)}",
            flush=True,
        )
        odd = {
            k: (b["status"], (b.get("error") or "")[:140])
            for k, b in rec["per_block"].items()
            if b["status"] != "optimized"
        }
        if odd:
            print(f"  non-optimized: {odd}", flush=True)
    except Exception as e:
        rec["opt_s"] = round(time.time() - t0, 1)
        rec["opt_error"] = f"{type(e).__name__}: {e}"
        print(f"  optimize FAILED: {rec['opt_error']}", flush=True)

    if opt_mod is not None:
        opt_mod = opt_mod.to(dev).eval()
        with torch.no_grad():
            rr = ref_model(images[0])
            oo = opt_mod(images[0])
            rr = rr[0] if isinstance(rr, (list, tuple)) else rr
            oo = oo[0] if isinstance(oo, (list, tuple)) else oo
            vr = verify_equiv(rr, oo)
        rec["fresh_rel"] = vr.max_rel
        rec["fresh_passed"] = vr.passed
        pe = _proxy_eval(ref_model, opt_mod, images)
        rec["proxy"] = pe
        print(
            f"  fresh rel={vr.max_rel:.3e}  "
            f"mask max_abs={pe['max_abs']:.3e} "
            f"max_rel={pe['max_rel']:.3e} IoU_min={pe['min_iou']:.4f}",
            flush=True,
        )

    # Timing arms — the 3.7 GB card holds ref + ONE delivered copy at
    # a time; the shared inductor baseline is measured once in main
    # (``args.inductor_timing``) and stamped into every record.
    rec["timing"] = {}
    if args.inductor_timing:
        rec["timing"]["inductor"] = args.inductor_timing
    arms = {"eager": ref_model}
    if opt_mod is not None:
        arms["catopt"] = opt_mod
    for name, mod in arms.items():
        if mod is None:
            continue
        mod.to(dev)
        r = _bench(mod, x_dev, int(args.reps))
        rec["timing"][name] = r
        if mod is not ref_model:
            mod.to("cpu")
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"  {name:>16}: {r['us']:9.1f}us  peak {r['peak_mb']:7.1f}MB",
            flush=True,
        )
    if opt_mod is not None and args.catopt_inductor:
        ref_model.to("cpu")  # headroom for the compiled extra copy
        oc, st_ = _try_compile(
            opt_mod, x_dev, float(args.compile_timeout)
        )
        rec["catopt_ind_status"] = st_
        if oc is not None:
            r = _bench(oc, x_dev, int(args.reps))
            rec["timing"]["catopt+inductor"] = r
            print(
                f"  {'catopt+inductor':>16}: {r['us']:9.1f}us  "
                f"peak {r['peak_mb']:7.1f}MB",
                flush=True,
            )
        ref_model.to(dev)
        print(f"  catopt+inductor: {st_}", flush=True)
    te = rec["timing"].get("eager", {}).get("us")
    tc = rec["timing"].get("catopt", {}).get("us")
    ti = rec["timing"].get("inductor", {}).get("us")
    if tc and te:
        rec["speedup_vs_eager"] = te / tc
    if tc and ti:
        rec["speedup_vs_inductor"] = ti / tc
    return rec


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the budget sweep on the real BiRefNet checkpoint."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument(
        "--budgets",
        type=str,
        default=_BUDGET_DEFAULT,
        help="comma list; 'none'/'exact' = exact-search baseline",
    )
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--max-iterations", type=int, default=100)
    ap.add_argument("--max-enodes", type=int, default=100_000)
    ap.add_argument("--compile-timeout", type=float, default=180.0)
    ap.add_argument("--no-inductor", action="store_true")
    ap.add_argument("--catopt-inductor", action="store_true")
    ap.add_argument("--full-ir-census", action="store_true")
    ap.add_argument(
        "--out", default="bench/results/birefnet_bounded.json"
    )
    args = ap.parse_args(argv)
    sys.setrecursionlimit(400_000)

    # Allocator/inductor env — set before the first CUDA tensor or
    # compile, so a plain ``import torch`` at module top is safe.
    # expandable_segments: on the 3.7 GB card fragmentation, not
    # footprint, is what OOMs the two-model timing loop.
    # COMPILE_THREADS=2: the default 16-worker compile pool stays
    # resident after the baseline and OOM-kills the next cell
    # (measured, not theoretical).
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("  --device cuda but CUDA is unavailable; using cpu")
        dev = torch.device("cpu")
    else:
        dev = torch.device(args.device)
    avail_gb = (
        __import__("os").sysconf("SC_AVPHYS_PAGES")
        * __import__("os").sysconf("SC_PAGE_SIZE")
        / 2**30
    )
    if avail_gb < 1.5:
        print(f"  ABORT: only {avail_gb:.1f}G RAM available (<1.5G)")
        return 2

    torch.manual_seed(0)
    model = _load_birefnet()
    n_params = sum(p.numel() for p in model.parameters())
    x_cpu = torch.randn(1, 3, args.res, args.res)
    print(
        f"birefnet_bounded — {_HF_REPO} {n_params / 1e6:.0f}M params "
        f"res={args.res} device={dev} budgets={args.budgets}",
        flush=True,
    )

    # -- structural census: the decoder BN question -------------------
    print("\n=== decoder IR census ===", flush=True)
    census = _decoder_census(model, x_cpu)
    if "error" in census:
        print(f"  decoder export failed: {census['error']}", flush=True)
    else:
        print(
            f"  {census['n_enodes']} enodes; "
            f"batch_norm nodes: {census['batch_norm_nodes']}",
            flush=True,
        )
        print(f"  ops: {census['ops']}", flush=True)
        print(
            f"  BatchNorm2d modules in tree: {len(census['bn_modules'])}",
            flush=True,
        )
        for n_ in census["bn_modules"]:
            print(f"    {n_}", flush=True)
    full_census = None
    if args.full_ir_census:
        print("\n=== full-model IR census ===", flush=True)
        try:
            full_census = _full_census(model, x_cpu)
            print(
                f"  {full_census['n_enodes']} enodes; "
                f"batch_norm: {full_census['batch_norm_nodes']}",
                flush=True,
            )
            print(f"  ops: {full_census['ops']}", flush=True)
        except Exception as e:
            full_census = {"error": f"{type(e).__name__}: {e}"}
            print(f"  full export failed: {full_census['error']}")

    # -- held-out images + a device-resident reference ------------------
    # ``model`` stays on CPU as the deepcopy master (per-cell search
    # copies then cost a host memcpy, not a GPU round-trip); the
    # device reference is a separate resident.
    ref_model = copy.deepcopy(model).to(dev).eval()
    x_dev = x_cpu.to(dev)
    images = _heldout_images(args.res, args.images, dev)

    # -- inductor baseline: compiled + timed ONCE, shared across cells --
    args.inductor_timing = None
    if not args.no_inductor:
        inductor_mod, ind_status = _try_compile(
            copy.deepcopy(ref_model), x_dev, float(args.compile_timeout)
        )
        print(f"\ninductor baseline: {ind_status}", flush=True)
        if inductor_mod is not None:
            r = _bench(inductor_mod, x_dev, int(args.reps))
            args.inductor_timing = r
            print(
                f"  {'inductor':>16}: {r['us']:9.1f}us  "
                f"peak {r['peak_mb']:7.1f}MB",
                flush=True,
            )
        del inductor_mod
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    budgets: list[tuple[str, float | None]] = []
    for tok in args.budgets.split(","):
        tok = tok.strip()
        if tok:
            budgets.append(
                (
                    tok,
                    None
                    if tok.lower() in ("none", "exact")
                    else float(tok),
                )
            )

    t0 = time.perf_counter()
    recs = []
    for label, budget in budgets:
        # Fresh CPU-resident search copy per cell — the delivered
        # module shares storage with its search model, so reusing one
        # copy across cells would let a previous cell's .to(dev) move
        # weights out from under the next search.
        search_model = copy.deepcopy(model).to("cpu").eval()
        rec = run_budget_cell(
            search_model,
            ref_model,
            x_cpu,
            x_dev,
            budget,
            label,
            args,
            dev,
            images,
        )
        del search_model
        recs.append(rec)
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        # Flush after every cell — a late-cell OOM kill must not
        # take the earlier cells' ledgers with it (measured: the
        # loose-budget dedup clustering is what grows).
        _flush_evidence(args, n_params, dev, census, full_census, recs)

    # -- sweep verdict -------------------------------------------------
    print("\n=== sweep ===", flush=True)
    hdr = (
        f"{'budget':>8} | {'opt s':>7} | {'offers':>6} | "
        f"{'deliv':>5} | {'bound':>9} | {'fresh_rel':>9} | "
        f"{'IoU min':>7} | {'catopt us':>10} | {'vs eager':>8} | "
        f"{'vs ind':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for rec in recs:
        t = rec.get("timing") or {}
        e = (t.get("eager") or {}).get("us")
        i = (t.get("inductor") or {}).get("us")
        c = (t.get("catopt") or {}).get("us")
        p = rec.get("proxy") or {}
        print(
            f"{rec['budget']:>8} | {rec.get('opt_s', 0):>7} | "
            f"{rec.get('n_bounded_offers', 0):>6} | "
            f"{rec.get('n_bounds_delivered', 0):>5} | "
            f"{rec.get('bound_total_max', 0.0):>9.2e} | "
            f"{(rec.get('fresh_rel') or 0.0):>9.2e} | "
            f"{(p.get('min_iou') or 0.0):>7.4f} | "
            + (f"{c:>10.1f} | " if c else f"{'—':>10} | ")
            + (f"{e / c:>7.3f}x | " if (c and e) else f"{'—':>8} | ")
            + (f"{i / c:>7.3f}x" if (c and i) else f"{'—':>8}")
        )
        if rec.get("opt_error"):
            verdict = f"optimize failed: {rec['opt_error']}"
        elif rec.get("bound_declines"):
            d = rec["bound_declines"][0]
            verdict = (
                f"bounded member delivered on {d['block']} but "
                f"DECLINED by the bound gate: measured rel "
                f"{d['measured_rel']:.2e} > propagated bound "
                f"{d['accepted_bound']:.2e}"
            )
        elif rec.get("n_bounds_delivered", 0) == 0:
            verdict = (
                "no bounded member delivered — budget changes nothing"
            )
            if rec.get("n_bounded_offers", 0) == 0:
                verdict = (
                    "no bounded offer even qualified — no projection "
                    "weight lies within budget of a structural special"
                )
        else:
            verdict = (
                f"{rec['n_bounds_delivered']} bounded members, "
                f"bound {rec['bound_total_max']:.2e}, "
                f"speedup {rec.get('speedup_vs_eager', 0.0):.3f}x, "
                f"IoU {p.get('min_iou', 0.0):.4f}"
            )
        rec["verdict"] = verdict
        print(f"{'':>8} | verdict: {verdict}")
    print("-" * len(hdr))
    print(
        f"  total wall time {time.perf_counter() - t0:.1f}s", flush=True
    )

    _flush_evidence(args, n_params, dev, census, full_census, recs)
    print(f"  evidence → {args.out}", flush=True)
    return 0


def _flush_evidence(
    args, n_params, dev, census, full_census, recs
) -> None:
    """Write the JSON payload accumulated so far (idempotent)."""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _HF_REPO,
        "params": n_params,
        "res": args.res,
        "device": str(dev),
        "decoder_census": census,
        "full_census": full_census,
        "records": recs,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
