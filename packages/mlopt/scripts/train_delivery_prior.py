"""Train the delivery prior — bake ``DELIVERY_PLAYER_WEIGHTS``.

The deliver board's ``linear*`` arms ship cold
(``mlopt.weights.DELIVERY_PLAYER_WEIGHTS == {}``).  This driver is
the kernelopt ``train_kernel_prior.py`` recipe adapted to delivery:

* **cases** — each ``--ref`` is lifted once (adapter ``load`` →
  ``Optimizer.search`` → eager ``lower(verify=…)``) into a
  :class:`mlopt.deliver_train.DeliveryCase`; the expensive lift is
  amortised over every imitation epoch and REINFORCE epoch;
* **oracle** — one scratch-board drain per case measures every
  eligible config through the real referee; the expert line is
  ``[argmin_config, "stop"]`` (or bare ``stop`` when eager wins);
* **train** — ``imitate_policy`` warm-start (oracle-positive cases
  only), then REINFORCE (``train_policy``) over the case stream;
  reward is the terminal committed speedup minus ``MOVE_COST`` per
  spent move;
* **bake** — prints the ``DELIVERY_PLAYER_WEIGHTS`` literal to paste
  into ``src/mlopt/weights.py`` (the ``KERNEL_PLAYER_WEIGHTS``
  convention), plus the held-out arm table when ``--eval-ref``
  cases are given.

Per-ref adapter options ride an ``@`` suffix::

    --ref "kokoro:hexgrad/kokoro@arm=decoder-nc,dtype=fp32"

— merged over the global ``--adapter-opt`` pairs.  Global opts are
the common case (``--adapter-opt device=cuda`` reaches every load).

Real training measures on the GPU — run it where the encoder
benchmarks are not.  Cover the arms where runners actually pay —
whisper *encoder* steps and the deterministic kokoro submodules.
``f0n`` lifts clean; plain ``textenc`` declines at lift
(``pack_padded_sequence`` CPU-lengths are not ``torch.export``-
able) but ``textenc-p`` — the pack→lstm→pad sites wrapped as
``custom_op`` leaves — exports and verifies (bench: diff 0.0);
the ``decoder``/``decoder-nc``/``full`` arms are
stochastic — ``SineGen`` draws in-graph — so they can never clear
``verify=True`` and belong only under ``--no-verify``:

    .venv/bin/python scripts/train_delivery_prior.py \
        --ref "whisper:openai/whisper-tiny@arm=encoder" \
        --ref "whisper:openai/whisper-base@arm=encoder" \
        --ref "kokoro:hexgrad/Kokoro-82M@arm=f0n" \
        --eval-ref "whisper:openai/whisper-small@arm=encoder" \
        --adapter-opt device=cuda --epochs 8

Decoder-step cases are kept honest by ``WIN_MARGIN`` (sub-2%
"wins" are bench noise and score 1.0, same as eager) — include
them, but expect most to teach ``stop``:

    --ref whisper:openai/whisper-tiny --ref whisper:openai/whisper-base

CPU smoke (tiny iters, still real lifts):

    .venv/bin/python scripts/train_delivery_prior.py \
        --ref whisper:openai/whisper-tiny --epochs 1 \
        --imitate-epochs 1 --adapter-opt device=cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mlopt.deliver_train import (
    BENCH_ITERS,
    BUDGET,
    EPOCHS,
    IMITATE_EPOCHS,
    SEED,
    DeliveryCase,
    _block_means,
    baked_table,
    cold_train,
    held_out,
    train,
    weights_literal,
)

__all__ = ["build_case", "main", "parse_ref"]


def _coerce(value: str) -> Any:
    """``--adapter-opt`` coercion: bool/int/float/str (cli parity)."""
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_ref(spec: str) -> tuple[str, dict[str, Any]]:
    """Split ``REF@k=v,k=v`` into ``(ref, opts)``.

    The ``@`` suffix carries per-case adapter options; global
    ``--adapter-opt`` pairs merge underneath (per-ref wins).
    """
    ref, _, tail = spec.partition("@")
    opts: dict[str, Any] = {}
    if tail:
        # Split on commas at bracket depth 0 — JSON values
        # (``leaves={"x":[32,64]}``) carry commas of their own.
        pairs: list[str] = []
        depth = 0
        start = 0
        for i, ch in enumerate(tail):
            if ch in "{[(":
                depth += 1
            elif ch in "}])":
                depth -= 1
            elif ch == "," and depth == 0:
                pairs.append(tail[start:i])
                start = i + 1
        pairs.append(tail[start:])
        for pair in pairs:
            key, sep, raw = pair.partition("=")
            if not sep or not key:
                raise ValueError(f"--ref option expects key=value, got {pair!r} in {spec!r}")
            opts[key.strip()] = _coerce(raw.strip())
    return ref, opts


def _adapter_opts(pairs: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--adapter-opt key=value`` args."""
    opts: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"--adapter-opt expects key=value, got {pair!r}")
        opts[key.strip()] = _coerce(raw.strip())
    return opts


def build_case(
    ref: str,
    *,
    name: str | None = None,
    opts: dict[str, Any] | None = None,
    bench_iters: int = BENCH_ITERS,
    verify: bool = True,
    budget: int = BUDGET,
    max_iterations: int = 4,
) -> DeliveryCase:
    """Lift *ref* once into a :class:`DeliveryCase`.

    Adapter ``load`` → ``Optimizer.search`` → eager
    ``lower(verify=verify)`` — exactly the objects
    ``optimize_model`` hands ``deliver_game``, minus the delivery
    pass itself (the game *is* the delivery decision here).
    """
    from kit_torch.backend import TorchBackend
    from mlopt_orchestrator.optimize import Optimizer

    import mlopt.models.catalog  # noqa: F401 — populates the registry
    from mlopt.models import resolve_ref
    from mlopt.optimize import _eqsat_timeout_seconds  # noqa: PLC2701

    adapter = resolve_ref(ref)
    if adapter is None:
        raise ValueError(f"no adapter claims {ref!r} — run `mlopt models`")
    if not adapter.available:
        raise RuntimeError(f"adapter {adapter.name!r} unavailable: {adapter.unavailable_reason}")
    mspec = adapter.load(ref, **(opts or {}))
    # ``spec.source`` is the pre-lifted-IR port (``.hnl``/``onnx-rust``/
    # ``arch:`` adapters) — ``optimize_model`` hands it to ``Optimizer``
    # in place of the backend's TorchSource; ``None`` keeps torch-export.
    # ``$MLOPT_EQSAT_TIMEOUT`` reaches the trainer here the same way it
    # reaches ``optimize_model`` — without this the search bound stayed
    # at the 300 s default and larger cases could never be lifted.
    engine = None
    eqsat_timeout = _eqsat_timeout_seconds()
    if eqsat_timeout is not None:
        from functools import partial

        from mlopt_orchestrator.eqsat import EqsatEngine

        engine = partial(EqsatEngine, timeout=eqsat_timeout)
        print(f"   eqsat search timeout: {eqsat_timeout:g}s via MLOPT_EQSAT_TIMEOUT")
    optz = Optimizer(
        backend=TorchBackend(),
        source=getattr(mspec, "source", None),
        engine=engine,
    )
    res = optz.search(mspec.module, mspec.inputs, max_iterations=max_iterations)
    lowered = optz.lower(res, mspec.inputs, verify=verify)
    return DeliveryCase(
        name=name or ref,
        optz=optz,
        res=res,
        lowered_eager=lowered,
        inputs=mspec.inputs,
        meta=dict(mspec.meta or {}),
        bench_iters=bench_iters,
        verify=verify,
        budget=budget,
        call_kwargs=dict(getattr(mspec, "kwargs", None) or {}),
    )


def _move_inputs(case: DeliveryCase, device: str) -> None:
    """Post-build input placement — the ``census_cases.py --device`` move.

    Adapters without a ``device`` load opt (``hnl``/``arch``) bind CPU
    tensors; ``torch_eval`` moves params to the input device per call,
    so moving the probe inputs is the whole placement and the board
    then offers the CUDA configs honestly.
    """
    import torch

    def _mv(t: Any) -> Any:
        return t.to(device) if isinstance(t, torch.Tensor) else t

    ins = case.inputs
    case.inputs = tuple(_mv(t) for t in ins) if isinstance(ins, (tuple, list)) else _mv(ins)


def _corpus_manifest(path: str) -> list[dict[str, Any]]:
    """Read a ``corpus_v6.json``-style manifest — the raw entries."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = doc["cases"] if isinstance(doc, dict) else doc
    return list(entries)


def _corpus_cases(
    entries: list[dict[str, Any]],
    names: list[str],
    *,
    global_opts: dict[str, Any],
    bench_iters: int,
    verify: bool,
    budget: int,
    max_iterations: int,
) -> tuple[list[DeliveryCase], dict[int, str | None], dict[int, dict[str, dict[str, Any]]]]:
    """Build the named cases from corpus *entries*.

    Each entry carries ``name``/``ref``/``opts`` (already merged at
    census time) plus the persisted oracle answer key
    (``oracle_pick`` + ``drain`` records) and an optional
    ``device_override`` applied post-build exactly as
    ``census_cases.py --device`` did.  Returns the cases plus the
    per-index pick/drain maps :func:`mlopt.deliver_train.train`
    accepts — the measured drains are reused, not re-run.  Train and
    eval subsets are built by separate calls so every case is
    lifted exactly once and the train stream can be freed before
    the held-out cases exist (peak-RSS matters on small hosts).
    """
    by_name = {e["name"]: e for e in entries}
    cases: list[DeliveryCase] = []
    picks: dict[int, str | None] = {}
    drains: dict[int, dict[str, dict[str, Any]]] = {}
    for name in names:
        entry = by_name[name]
        merged = {**global_opts, **(entry.get("opts") or {})}
        print(f"== lifting {name} ({entry['ref']}, opts={merged or '—'}) ==", flush=True)
        case = build_case(
            entry["ref"],
            name=name,
            opts=merged,
            bench_iters=bench_iters,
            verify=verify,
            budget=budget,
            max_iterations=max_iterations,
        )
        device = entry.get("device_override")
        if device:
            _move_inputs(case, device)
        idx = len(cases)
        cases.append(case)
        picks[idx] = entry.get("oracle_pick")
        drains[idx] = dict(entry.get("drain") or {})
        n_enodes = getattr(case.res, "n_enodes", "?")
        print(f"   case {name}: n_enodes={n_enodes} pick={picks[idx]}", flush=True)
    return cases, picks, drains


def _build_cases(
    specs: list[str],
    *,
    global_opts: dict[str, Any],
    bench_iters: int,
    verify: bool,
    budget: int,
    max_iterations: int,
) -> list[DeliveryCase]:
    """Lift every ``REF[@opts]`` spec — prints one line per case."""
    cases: list[DeliveryCase] = []
    for spec in specs:
        ref, opts = parse_ref(spec)
        merged = {**global_opts, **opts}
        print(f"== lifting {ref} (opts={merged or '—'}) ==", flush=True)  # stdout-compat
        case = build_case(
            ref,
            opts=merged,
            bench_iters=bench_iters,
            verify=verify,
            budget=budget,
            max_iterations=max_iterations,
        )
        cases.append(case)
        n_enodes = getattr(case.res, "n_enodes", "?")
        print(f"   case {case.name}: n_enodes={n_enodes}")  # stdout-compat
    return cases


def main(argv: list[str] | None = None) -> int:
    """Train, print the curve, the literal, and the held-out table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ref",
        action="append",
        default=[],
        metavar="REF[@k=v,...]",
        help="train case — adapter ref, repeatable",
    )
    ap.add_argument(
        "--corpus",
        default=None,
        metavar="PATH",
        help="train from a census_cases.py --drain corpus JSON "
        "(case manifests + persisted oracle answer key) instead of "
        "--ref; mutually exclusive",
    )
    ap.add_argument(
        "--eval-name",
        action="append",
        default=[],
        metavar="NAME",
        help="corpus mode: held-out case name (repeatable) — every other non-skipped case trains",
    )
    ap.add_argument(
        "--skip-name",
        action="append",
        default=[],
        metavar="NAME",
        help="corpus mode: drop a case entirely (e.g. a known-poisoned drain), repeatable",
    )
    ap.add_argument(
        "--eval-ref",
        action="append",
        default=[],
        metavar="REF[@k=v,...]",
        help="held-out case — disjoint workloads, repeatable",
    )
    ap.add_argument(
        "--adapter-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="global adapter load option (e.g. device=cuda), repeatable",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--imitate-epochs", type=int, default=IMITATE_EPOCHS)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--bench-iters", type=int, default=BENCH_ITERS)
    ap.add_argument("--iters", type=int, default=4, help="search max_iterations")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument(
        "--cold-rl",
        action="store_true",
        help="also train a cold-REINFORCE-only control arm",
    )
    ap.add_argument(
        "--aggregate",
        choices=("mean", "median", "case-norm"),
        default="mean",
        help="held-out table collapse — 'mean' (shipped default), "
        "'median' (per-case median then median across cases), or "
        "'case-norm' (per-case ratio to the best arm's median, then "
        "mean — the de-noised v12 metric)",
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)

    if args.corpus and (args.ref or args.eval_ref):
        ap.error("--corpus is mutually exclusive with --ref/--eval-ref")
    if not args.ref and not args.corpus:
        ap.error("at least one --ref or --corpus is required (the train case stream)")
    if args.eval_name and not args.corpus:
        ap.error("--eval-name only applies to --corpus mode")
    if args.skip_name and not args.corpus:
        ap.error("--skip-name only applies to --corpus mode")
    try:
        global_opts = _adapter_opts(args.adapter_opt)
    except ValueError as e:
        ap.error(str(e))

    oracle_picks: dict[int, str | None] | None = None
    oracle_drains: dict[int, dict[str, dict[str, Any]]] | None = None
    corpus_entries: list[dict[str, Any]] = []
    train_case_names: list[str] = []
    eval_case_names: list[str] = []
    if args.corpus:
        corpus_entries = _corpus_manifest(args.corpus)
        by_name = {e["name"]: e for e in corpus_entries}
        skip = set(args.skip_name)
        eval_set = set(args.eval_name)
        available = set(by_name) - skip
        unknown = (eval_set | skip) - set(by_name)
        if unknown:
            ap.error(f"names not in corpus: {sorted(unknown)}")
        bad_eval = eval_set - available
        if bad_eval:
            ap.error(f"--eval-name also --skip-name'd: {sorted(bad_eval)}")
        # Corpus order is preserved for both subsets.
        train_case_names = [n for n in by_name if n in available and n not in eval_set]
        eval_case_names = [n for n in by_name if n in eval_set]
        cases, oracle_picks, oracle_drains = _corpus_cases(
            corpus_entries,
            train_case_names,
            global_opts=global_opts,
            bench_iters=args.bench_iters,
            verify=not args.no_verify,
            budget=args.budget,
            max_iterations=args.iters,
        )
        print(  # stdout-compat
            f"== corpus split: {len(cases)} train / {len(eval_case_names)} held-out "
            f"({eval_case_names}) =="
        )
    else:
        cases = _build_cases(
            args.ref,
            global_opts=global_opts,
            bench_iters=args.bench_iters,
            verify=not args.no_verify,
            budget=args.budget,
            max_iterations=args.iters,
        )
        train_case_names = [c.name for c in cases]

    player, totals, imitation, picks, drains = train(
        cases,
        seed=args.seed,
        epochs=args.epochs,
        imitate_epochs=args.imitate_epochs,
        budget=args.budget,
        oracle_picks=oracle_picks,
        oracle_drains=oracle_drains,
    )
    cold_rl = None
    cold_totals: list[float] = []
    if args.cold_rl:
        cold_rl, cold_totals = cold_train(
            cases,
            seed=args.seed,
            epochs=args.epochs,
            budget=args.budget,
        )
    n_train_cases = len(cases)
    # Free the train stream before the held-out cases exist — every
    # DeliveryCase pins a full lifted module, and peak RSS is what
    # gets a long run earlyoom'd on a contended host.
    del cases
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — cache hygiene is best-effort
        pass  # nosec B110
    if args.corpus:
        eval_cases, _ep, _ed = _corpus_cases(
            corpus_entries,
            eval_case_names,
            global_opts=global_opts,
            bench_iters=args.bench_iters,
            verify=not args.no_verify,
            budget=args.budget,
            max_iterations=args.iters,
        )
    else:
        eval_cases = (
            _build_cases(
                args.eval_ref,
                global_opts=global_opts,
                bench_iters=args.bench_iters,
                verify=not args.no_verify,
                budget=args.budget,
                max_iterations=args.iters,
            )
            if args.eval_ref
            else []
        )
    table = baked_table(player)
    curve = _block_means(totals, args.blocks)
    print(  # stdout-compat
        f"== train delivery prior — seed {args.seed}, "
        f"{n_train_cases} cases x {args.epochs} epochs (+ "
        f"{args.imitate_epochs} imitation ep) =="
    )
    print(  # stdout-compat
        f"oracle picks: {picks}"
    )
    print(  # stdout-compat
        f"imitation: {imitation['imitated']}/{imitation['shown']} demonstrated moves cloned"
    )
    print(f"train block means: {curve}")  # stdout-compat
    if cold_totals:
        print(  # stdout-compat
            f"cold-rl block means: {_block_means(cold_totals, args.blocks)}"
        )
    held_errors: dict[str, str] = {}
    held_done: dict[str, int] = {}
    if eval_cases:
        try:
            held, held_errors, held_done = held_out(
                player,
                eval_cases,
                seed=args.seed,
                budget=args.budget,
                cold_rl=cold_rl,
                aggregate=args.aggregate,
            )
        except Exception as e:  # noqa: BLE001 — never lose the weights to eval
            held = {}
            held_errors["*"] = f"{type(e).__name__}: {e}"
            print(f"== held-out failed: {held_errors['*'].splitlines()[0]} ==")  # stdout-compat
        print("== held-out ==")  # stdout-compat
        for name, reward in held.items():
            n = held_done.get(name, 0)
            suffix = ", eval error)" if name in held_errors else ")"
            partial = f" ({n}/{len(eval_cases)} cells{suffix}"
            print(f"  {name:16s} reward={reward:8.3f}{partial}")  # stdout-compat
        for name, err in held_errors.items():
            print(f"  {name:16s} eval failed: {err.splitlines()[0]}")  # stdout-compat
    else:
        print("== held-out skipped (no eval cases) ==")  # stdout-compat
    print("== baked snapshot — paste into src/mlopt/weights.py ==")  # stdout-compat
    print(weights_literal(table))  # stdout-compat
    if args.json:
        payload: dict[str, Any] = {
            "corpus": args.corpus,
            "aggregate": args.aggregate,
            "held_out": sorted(eval_set) if args.corpus else [c.name for c in eval_cases],
            "skipped": sorted(args.skip_name),
            "weights": table,
            "curve": curve,
            "imitation": imitation,
            "oracle_picks": {str(k): v for k, v in picks.items()},
            # The persisted answer key — every config's measured
            # min_ms/med_ms/verified/reason per case (autopsy §4).
            "oracle_drains": {str(k): v for k, v in drains.items()},
            "cold_curve": (_block_means(cold_totals, args.blocks) if cold_totals else None),
            "held_out_cells": held_done,
            "held_out_errors": held_errors,
            "cases": train_case_names,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
