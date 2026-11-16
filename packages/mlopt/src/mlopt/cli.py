"""``mlopt`` / ``python -m mlopt`` — product CLI.

Commands mirror the ``hnl-ext-mlopt`` extension surface: ``models``
lists the adapter catalog, ``optimize`` runs the verified pipeline.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _cmd_models(_args: argparse.Namespace) -> int:
    import mlopt.models.catalog  # noqa: F401 — populates the registry
    from mlopt.models import list_adapters

    for name, adapter in sorted(list_adapters().items()):
        status = "ready" if adapter.available else f"pending — {adapter.unavailable_reason}"
        print(f"{name:<10} {status}\n{'':<10} {adapter.description}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from mlopt.doctor import diagnose

    print(diagnose(args.model))
    return 0


def _coerce_opt_value(raw: str) -> object:
    """``--adapter-opt`` value coercion: bool/int/float/str."""
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _adapter_opts(pairs: list[str] | None) -> dict[str, object]:
    """Parse repeated ``--adapter-opt key=value`` args into load kwargs."""
    opts: dict[str, object] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--adapter-opt expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--adapter-opt expects key=value, got {pair!r}")
        opts[key] = _coerce_opt_value(raw)
    return opts


def _load_spec(args: argparse.Namespace):
    """Shared adapter resolution + load for optimize/export.

    Returns ``(spec, 0)`` on success, ``(None, code)`` after printing
    the honest failure (2 = no/unavailable adapter, 1 = load).
    """
    import mlopt.models.catalog  # noqa: F401 — populates the registry
    from mlopt.models import resolve_ref

    # One ref semantics everywhere (resolve_ref): ``name:`` prefix,
    # bare adapter names, claims() for other bare refs; whitespace
    # stripped — the same ref doctor reports on.
    ref = args.model.strip()
    adapter = resolve_ref(ref)
    if adapter is None:
        print(f"mlopt: no adapter claims `{ref}` — run `mlopt models`", file=sys.stderr)
        return None, 2
    if not adapter.available:
        reason = adapter.unavailable_reason
        print(f"mlopt: adapter `{adapter.name}` unavailable: {reason}", file=sys.stderr)
        return None, 2
    try:
        opts = _adapter_opts(getattr(args, "adapter_opt", None))
    except ValueError as e:
        print(f"mlopt: {e}", file=sys.stderr)
        return None, 2
    try:
        return adapter.load(ref, backend=args.backend, **opts), 0
    except Exception as e:  # noqa: BLE001
        print(f"mlopt: load failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None, 1


def _cmd_optimize(args: argparse.Namespace) -> int:
    from mlopt.optimize import optimize_model

    spec, code = _load_spec(args)
    if spec is None:
        return code

    lowered, report = optimize_model(
        spec,
        backend=args.backend,
        rules=args.rules,
        max_iterations=args.iters,
        verify=not args.no_verify,
        deliver=args.deliver,
        bench_iters=args.bench_iters,
        deliver_budget=args.deliver_budget or None,
        deliver_player=args.deliver_player,
        deliver_heuristic=args.deliver_heuristic,
    )
    print(report.render())
    if lowered is None:
        return 1
    if report.verified is False:
        print("mlopt: verify gate FAILED — artifact not trustworthy", file=sys.stderr)
        return 1
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """``mlopt export`` — optimized term → ``.hnl`` over ``tensor.*``.

    Runs the same verified pipeline as ``optimize`` (``verify=True``
    is not optional here — the export artifact claims the verified
    term), then emits the *search* result's extracted term: the
    ``LowerResult`` carries only the delivered executor module, not a
    term, so ``SearchResult.term`` is the honest source (recorded on
    ``OptimizeReport.search_result``).  ``Param`` leaves bind to their
    ``param_values``; ``Var`` leaves become the exported function's
    parameters.
    """
    import re

    from mlopt.emit import UnsupportedOpError, emit_function, find_hnl_run
    from mlopt.optimize import optimize_model

    spec, code = _load_spec(args)
    if spec is None:
        return code

    lowered, report = optimize_model(
        spec,
        backend=args.backend,
        rules=args.rules,
        max_iterations=args.iters,
        verify=True,
    )
    print(report.render())
    if lowered is None or report.search_result is None:
        return 1
    if report.verified is False:
        print(
            "mlopt: verify gate FAILED — refusing to emit an unverified artifact",
            file=sys.stderr,
        )
        return 1

    res = report.search_result
    term = getattr(res, "term", None)
    if term is None:
        print("mlopt: export: search result carries no term", file=sys.stderr)
        return 1
    feed = dict(getattr(res, "param_values", None) or {})

    slug = re.sub(r"[^A-Za-z0-9]+", "-", args.model).strip("-").lower() or "model"
    out = args.out or f"{slug}.hnl"
    from pathlib import Path

    # --sidecar: large Param leaves emit tensor.load("<basename>",
    # "<leaf>") and their values land in <out>.safetensors beside the
    # artifact (emit writes the file only when a leaf sidecars).
    sidecar = None
    emit_kw: dict[str, Any] = {}
    if args.sidecar:
        sidecar = Path(out).with_suffix(".safetensors")
        emit_kw = {"sidecar": sidecar, "sidecar_dtype": args.sidecar_dtype}
    try:
        src = emit_function(term, feed, name="optimized", **emit_kw)
    except (UnsupportedOpError, ValueError) as e:
        print(f"mlopt: export declined: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    try:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(src + "\n")
    except OSError as e:
        print(f"mlopt: export: cannot write {out}: {e}", file=sys.stderr)
        return 1
    print(f"export:   {out}")
    if sidecar is not None:
        if "tensor.load(" in src:
            size = sidecar.stat().st_size if sidecar.exists() else -1
            print(f"sidecar:  {sidecar} ({size} bytes)")
        else:
            # No Param leaf reached the size gate — nothing referenced.
            print(
                f"sidecar:  {sidecar} not written — every Param leaf is below the inline threshold"
            )
            sidecar = None

    if args.check:
        binary = find_hnl_run()
        if binary is None:
            print("check:    skipped — hnl-run not found (set $MLOPT_HNL_RUN)")
            return 0
        from mlopt.emit import KitRunner

        try:
            # tensor_load_dir = the sidecar's directory: the emitted
            # tensor.load paths are basenames resolved via
            # $TENSOR_LOAD_DIR (hnl-run - has no artifact directory).
            KitRunner(
                binary,
                tensor_load_dir=sidecar.parent if sidecar is not None else None,
            ).run(src)
        except Exception as e:  # noqa: BLE001 — report honestly
            print(f"check:    hnl-run rejected the artifact: {e}", file=sys.stderr)
            return 1
        print(f"check:    hnl-run accepted ({binary})")

    if args.verify_run:
        return _verify_run_export(
            term,
            feed,
            spec,
            res,
            sidecar=sidecar,
            sidecar_dtype=args.sidecar_dtype,
        )
    return 0


def _prelifted_ir_params(spec: Any, search_result: Any) -> Any:
    """Resolve ``(ir, param_values)`` for a data-carrier ``spec.module``.

    ``onnx-rust``/``hnl``/``arch`` specs carry the lifted term
    (``OnnxLift`` / ``HnlLift`` / ``ArchLift``), not a callable — but
    ``lower(verify=True)`` already proved the gate's reference is
    ``sink.lower(source_ir, params)``, so lowering the search result's
    *source* IR through ``ir_to_torch_module`` IS the honest torch
    reference here.  The pair comes from ``SearchResult.ir``/
    ``param_values`` first, then the spec's ``Source`` port as
    fallback (the pre-lifted sources answer ``to_ir`` with the same
    pair).  Returns ``(ir, params)`` or ``None`` — the decline is
    printed here; per-dtype lowering is the caller's job so an f64
    attempt can fall back to native without re-resolving.
    """
    ir = getattr(search_result, "ir", None) if search_result is not None else None
    params = getattr(search_result, "param_values", None) if search_result is not None else None
    if ir is None:
        to_ir = getattr(getattr(spec, "source", None), "to_ir", None)
        if callable(to_ir):
            try:
                ir, params = to_ir(getattr(spec, "module", None), getattr(spec, "inputs", ()))
            except Exception as e:  # noqa: BLE001 — surface the source's own error
                print(
                    f"verify-run: declined — spec.source.to_ir failed: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                return None
    if ir is None:
        print(
            "verify-run: declined — spec.module is not callable and no "
            "source IR is available for the torch reference",
            file=sys.stderr,
        )
        return None
    return ir, dict(params or {})


def _f64_args(xs: Any) -> tuple[Any, ...]:
    """Cast floating-point torch tensors to float64; everything else untouched.

    Integer/bool operands are semantic (indices, masks) and complex
    inputs would lose their imaginary part — only real-floating
    tensors upcast.
    """
    import torch as _t

    return tuple(
        x.to(_t.float64) if isinstance(x, _t.Tensor) and x.is_floating_point() else x for x in xs
    )


def _float_param_bytes(module: Any) -> int | None:
    """Estimated bytes of a module's floating-point parameters.

    ``None`` when the module does not expose ``parameters()`` — the
    f64-reference attempt is then skipped (native dtype + scaled
    tolerance instead).
    """
    try:
        return sum(
            int(p.numel()) * int(p.element_size())
            for p in module.parameters()
            if p.is_floating_point()
        )
    except Exception:  # noqa: BLE001 — not a torch module / exotic params
        return None


def _verify_run_tolerance(dtype: Any, payload_dtype: str = "F64") -> tuple[float, float, str]:
    """``(atol, rtol, rationale)`` for the verify-run numeric compare.

    Two dtypes bound the honest tolerance — the *reference* dtype and
    the *payload* dtype of the weights the host program actually
    computes from — and the gate takes the coarser of the two (it
    widens, never tightens, and the report line names both):

    *reference* — the emitted ``tensor.*`` program evaluates dense
    f64, so the gate upcasts the reference to f64 where it can (param
    values + floating-point inputs, or ``Module.double()`` for callable
    specs) and compares at a tight bound.  When the reference can only
    run in its native dtype — a callable module too large to double on
    this host, or a dtype the upcast cannot reach — the tolerance
    scales with that dtype's epsilon: ``512·eps`` covers reassociation
    noise accumulated over a deep graph (f32 → ~6.1e-5) without
    approaching a real bug's magnitude; floored at the historical
    1e-6 and capped at 1e-2 for sub-f32 floats.  Non-floating outputs
    compare exactly.

    *payload* — ``"F32"`` means the safetensors sidecar rounded every
    sidecar'd ``Param`` leaf to f32 on write, so the host computes
    from f32-precision weights while an f64 reference holds the
    unrounded values; the f32-scaled bound is then the honest floor
    (weight quantization ~eps_f32 relative is indistinguishable from
    compute reassociation at that scale).  ``"F64"`` (and no sidecar —
    inline ``from_list`` literals are f64) adds nothing.
    """
    try:
        import torch as _t
    except Exception:  # noqa: BLE001 — no torch, no dtype to scale by
        return 1e-6, 1e-5, "torch unavailable — historical f32 bound"
    if dtype == _t.float64:
        atol, rtol = 1e-9, 1e-9
        why = "f64 reference vs dense-f64 host — tight bound"
    else:
        try:
            eps = float(_t.finfo(dtype).eps)
        except Exception:  # noqa: BLE001 — int/bool/other dtypes have no finfo
            atol, rtol = 0.0, 0.0
            why = f"{dtype} reference — exact compare"
        else:
            tol = min(max(512.0 * eps, 1e-6), 1e-2)
            atol = rtol = tol
            why = f"{dtype} reference — 512·eps({dtype})={tol:.2e} reassociation bound"
    if payload_dtype == "F32":
        ptol = min(max(512.0 * float(_t.finfo(_t.float32).eps), 1e-6), 1e-2)
        atol, rtol = max(atol, ptol), max(rtol, ptol)
        why += (
            f"; F32 sidecar payload — Param weights quantized to f32, bound 512·eps(f32)={ptol:.2e}"
        )
    return atol, rtol, why


def _verify_run_export(
    term: Any,
    feed: dict[str, Any],
    spec: Any,
    search_result: Any = None,
    *,
    sidecar: Any = None,
    sidecar_dtype: str = "F64",
) -> int:
    """``mlopt export --verify-run`` — numeric gate on the artifact.

    Emits the SAME term as a runnable ``do`` program (every leaf fed),
    evaluates it through ``hnl-run``, and compares the decoded tensor
    against the torch reference on ``spec.inputs``.  The reference is
    ``spec.module(*inputs)`` when the spec carries a callable; for
    pre-lifted data carriers (``onnx-rust``/``hnl`` — ``spec.module``
    is a term payload, not a callable) it is the source IR lowered
    through ``ir_to_torch_module`` — the same reference
    ``lower(verify=True)`` compared against.  Var leaves bind to
    ``spec.inputs`` by NAME through the source IR's input order when
    it is recoverable (``ir.inputs`` is built in the same order as
    ``spec.inputs``), falling back to first-use positional order —
    a wrong binding fails the numeric compare, it cannot pass
    silently — and only a single-tensor output is supported —
    anything else declines.

    The torch reference runs in float64 where it can (param values +
    floating-point inputs upcast, or ``Module.double()`` on a deepcopy
    for callable specs under a size budget) because the host
    ``tensor.*`` program evaluates dense f64 — an f32 reference would
    inject ~1e-5 probe noise the gate would have to tolerate.  A
    native-dtype fallback keeps the gate alive for modules too large
    to double; the compare tolerance then scales with the reference
    dtype's epsilon (see :func:`_verify_run_tolerance`) and the report
    line names the bound and its rationale.  When *sidecar* is a real
    file, its *sidecar_dtype* is the payload dtype the tolerance also
    accounts for: an ``"F32"`` sidecar quantizes the ``Param`` weights
    on write, so the bound widens to the f32 scale rather than
    pretending the host computed from f64 weights.
    """
    import sys

    from mlopt.emit import KitRunner, UnsupportedOpError, emit_program, leaf_kinds

    vars_in_term = [n for n, k in leaf_kinds(term) if k == "var"]
    inputs = tuple(getattr(spec, "inputs", ()) or ())
    if len(vars_in_term) != len(inputs):
        print(
            "verify-run: declined — "
            f"{len(vars_in_term)} Var leaves vs {len(inputs)} probe inputs; "
            "positional binding is only honest at equal arity",
            file=sys.stderr,
        )
        return 1
    run_feed = dict(feed)
    # Bind by NAME when the source IR's input order is recoverable:
    # ``spec.inputs`` follows the adapter's *declared* order, which
    # need not match the term's first-use (``leaf_kinds``) order — a
    # positional zip then feeds each Var the wrong probe (the
    # ``arch:whisper-tiny-full`` conv1d rank failure was exactly
    # this: ``input_features`` received the ids tensor).  ``ir.inputs``
    # is built in the same order as ``spec.inputs``, so its names are
    # the honest key; anything short of a full name match falls back
    # to the positional contract.
    ir = getattr(search_result, "ir", None) if search_result is not None else None
    if ir is None:
        to_ir = getattr(getattr(spec, "source", None), "to_ir", None)
        if callable(to_ir):
            try:
                ir, _params = to_ir(getattr(spec, "module", None), inputs)
            except Exception:  # noqa: BLE001 — name binding is best-effort
                ir = None
    in_names = [
        n
        for v in (getattr(ir, "inputs", None) or ())
        if isinstance((n := getattr(v, "name", None)), str)
    ]
    if len(in_names) == len(inputs) and set(in_names) == set(vars_in_term):
        for name, val in zip(in_names, inputs, strict=True):
            run_feed[name] = val
    else:
        for name, val in zip(vars_in_term, inputs, strict=True):
            run_feed[name] = val

    # Large Var probes ride the safetensors channel too: an inline
    # ``from_list`` literal the size of whisper's ``encoder_hidden_states``
    # probe (576k f64) is what got ``hnl-run`` signal-killed on
    # ``arch:whisper-tiny-decoder --verify-run``.  Param leaves are
    # untouched — they keep the artifact's own ``tensor.load`` spelling;
    # only Var feeds at/above the sidecar element gate become
    # ``tensor.load("<feed>", "<var>")`` entries in a gate-scratch
    # safetensors (removed after the run).
    import json as _json
    import tempfile
    from pathlib import Path

    from mlopt.emit import _SIDECAR_MIN_ELEMS, _as_array, write_safetensors

    var_feed: dict[str, Any] = {}
    for name in vars_in_term:
        arr = _as_array(run_feed.get(name))
        if arr is not None and len(arr[1]) >= _SIDECAR_MIN_ELEMS:
            var_feed[name] = arr
    feed_path: Path | None = None
    feed_dir: Path | None = None
    tensor_load_dir = sidecar.parent if sidecar is not None else None
    if var_feed:
        if sidecar is not None:
            feed_dir = Path(sidecar).parent
            feed_path = feed_dir / f"{Path(sidecar).stem}.verify-inputs.safetensors"
        else:
            feed_dir = Path(tempfile.mkdtemp(prefix="mlopt-verify-feed-"))
            feed_path = feed_dir / "inputs.safetensors"
        for name in var_feed:
            run_feed[name] = {
                "hnl": (f"tensor.load({_json.dumps(feed_path.name)}, {_json.dumps(name)})")
            }
        tensor_load_dir = feed_dir
    try:
        # sidecar= re-emits the SAME tensor.load spelling the artifact
        # carries (deterministic rewrite of the same file) so the gate
        # exercises the shipped weight channel, not just the term.
        src = emit_program(
            term,
            run_feed,
            sidecar=sidecar,
            sidecar_dtype=sidecar_dtype,
        )
    except UnsupportedOpError as e:
        print(f"verify-run: declined — {e}", file=sys.stderr)
        return 1
    if var_feed:
        assert feed_path is not None  # set in the var_feed branch above  # nosec B101
        try:
            write_safetensors(feed_path, var_feed)
        except Exception as e:  # noqa: BLE001 — feed write failure is a gate failure
            print(f"verify-run: feed safetensors write failed: {e}", file=sys.stderr)
            return 1
    try:
        runner = KitRunner(tensor_load_dir=tensor_load_dir)
    except FileNotFoundError as e:
        print(f"verify-run: declined — {e}", file=sys.stderr)
        return 1
    try:
        try:
            out = runner.run(src)
        except Exception as e:  # noqa: BLE001 — hnl-run failure is a gate failure
            print(f"verify-run: hnl-run rejected the program: {e}", file=sys.stderr)
            return 1
    finally:
        if feed_path is not None:
            import contextlib

            with contextlib.suppress(OSError):
                feed_path.unlink()
            if sidecar is None and feed_dir is not None:
                with contextlib.suppress(OSError):
                    feed_dir.rmdir()
    wire = out.get("__tensor__") if isinstance(out, dict) else None
    if wire is None:
        print(
            "verify-run: declined — the program's result is not a single "
            f"tensor ({out!r}); only tensor outputs are compared",
            file=sys.stderr,
        )
        return 1

    try:
        import torch as _t
    except Exception as e:  # noqa: BLE001 — the reference is torch by contract
        print(f"verify-run: declined — torch unavailable: {e}", file=sys.stderr)
        return 1

    module = getattr(spec, "module", None)
    kwargs = dict(getattr(spec, "kwargs", None) or {})
    # Reference attempts, best-first: an f64 reference (matching the
    # host's dense-f64 eval) then the native-dtype one.  Each entry is
    # ``(label, caller)``; the first that returns a tensor wins and its
    # label/dtype drive the compare tolerance.
    ref_attempts: list[tuple[str, Any]] = []
    if module is not None and callable(module):
        # f64 first: deepcopy + .double() so the spec's module is never
        # mutated, and only when the f64 weight set plausibly fits —
        # doubling a multi-GB module on an 11GB host is an OOM kill,
        # not a catchable error, so big modules skip straight to the
        # native reference (epsilon-scaled tolerance below).
        f64_bytes = _float_param_bytes(module)
        if f64_bytes is not None and 2 * f64_bytes <= (4 << 30):
            try:
                import copy as _copy

                mod64 = _copy.deepcopy(module)
                if hasattr(mod64, "double"):
                    mod64 = mod64.double()
                elif hasattr(mod64, "to"):
                    mod64 = mod64.to(dtype=_t.float64)
                else:
                    raise TypeError("module exposes no dtype cast")
                xs64 = _f64_args(inputs)
                ref_attempts.append(("f64", lambda: mod64(*xs64, **kwargs)))  # noqa: E731
            except Exception:  # noqa: BLE001 — f64 reference is best-effort
                pass  # nosec B110
        ref_attempts.append(("native", lambda: module(*inputs, **kwargs)))  # noqa: E731
    else:
        # Pre-lifted data carrier (onnx-rust/hnl/arch): spec.module is
        # the lifted-term payload, not a callable — rebuild the torch
        # reference from the source IR, the same lowering the
        # lower(verify=True) gate compared against.  f64 first: the
        # IRModule takes the param tensors' dtype, so upcasting
        # param_values + float inputs runs the whole graph in f64.
        if kwargs:
            print(
                "verify-run: declined — spec.kwargs cannot reach an IRModule reference",
                file=sys.stderr,
            )
            return 1
        pair = _prelifted_ir_params(spec, search_result)
        if pair is None:
            return 1
        ir, params = pair
        try:
            from kit_torch.torch_bridge import ir_to_torch_module
        except Exception as e:  # noqa: BLE001
            print(
                f"verify-run: declined — torch reference lowering failed: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            return 1

        def _lower(dtype: Any) -> Any:
            pv = params
            if dtype is not None:
                pv = {
                    k: (v.to(dtype) if isinstance(v, _t.Tensor) and v.is_floating_point() else v)
                    for k, v in params.items()
                }
            return ir_to_torch_module(ir, param_values=dict(pv))

        # Lazy thunks: lowering an IRModule folds weight-only subtrees
        # (a second full pass over the weight set), so the native
        # module is only built when the f64 attempt actually fails.
        for label, dtype, xs in (
            ("f64", _t.float64, _f64_args(inputs)),
            ("native", None, inputs),
        ):
            ref_attempts.append(
                (label, lambda d=dtype, x=xs: _lower(d)(*x))  # noqa: E731
            )

    ref_t = None
    ref_label = ""
    ref_err: Exception | None = None
    for label, caller in ref_attempts:
        try:
            ref = caller()
        except Exception as e:  # noqa: BLE001 — try the next dtype
            ref_err = e
            continue
        detach = getattr(ref, "detach", None)
        ref_t = detach() if callable(detach) else ref
        ref_label = label
        break
    if ref_t is None:
        print(f"verify-run: torch reference failed: {ref_err}", file=sys.stderr)
        return 1
    try:
        if not isinstance(ref_t, _t.Tensor):
            raise TypeError(f"reference output is {type(ref_t).__name__}, not a Tensor")
        ref_flat = ref_t.detach().reshape(-1).to("cpu", _t.float64).tolist()
    except Exception as e:  # noqa: BLE001
        print(f"verify-run: declined — torch output unusable: {e}", file=sys.stderr)
        return 1

    got_flat = list(wire.get("data") or [])
    if list(wire.get("shape") or []) != list(getattr(ref_t, "shape", ())):
        print(
            f"verify-run: FAILED — shape {wire.get('shape')} != torch {list(ref_t.shape)}",
            file=sys.stderr,
        )
        return 1
    if len(got_flat) != len(ref_flat):
        print(
            f"verify-run: FAILED — {len(got_flat)} host values vs {len(ref_flat)} torch values",
            file=sys.stderr,
        )
        return 1
    diffs = [abs(float(a) - float(b)) for a, b in zip(got_flat, ref_flat, strict=True)]
    max_abs = max(diffs) if diffs else 0.0
    # Payload dtype only matters when a sidecar file was actually
    # written: with sidecar=None every Param leaf emitted an inline
    # f64 ``from_list`` literal, so ``--sidecar-dtype`` alone widens
    # nothing (it is inert without --sidecar).
    payload_dtype = sidecar_dtype if sidecar is not None else "F64"
    atol, rtol, tol_why = _verify_run_tolerance(
        getattr(ref_t, "dtype", None), payload_dtype=payload_dtype
    )
    ref_dtype = getattr(ref_t, "dtype", "?")
    tol_note = (
        f"ref={ref_label} dtype={ref_dtype} payload={payload_dtype} "
        f"atol={atol:.1e} rtol={rtol:.1e} — {tol_why}"
    )
    bad = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(got_flat, ref_flat, strict=True))
        if abs(float(a) - float(b)) > atol + rtol * abs(float(b))
    ]
    if bad:
        i, a, b = bad[0]
        print(
            f"verify-run: FAILED — max_abs={max_abs:.3e} at flat[{i}] "
            f"(host={a!r} torch={b!r}); {len(bad)} mismatches; {tol_note}",
            file=sys.stderr,
        )
        return 1
    print(f"verify-run: PASS — max_abs={max_abs:.3e} over {len(ref_flat)} values; {tol_note}")
    return 0


def _cmd_stage(args: argparse.Namespace) -> int:
    """``mlopt stage <name>`` — the kit-mlopt JSON stdin/stdout protocol."""
    import json

    from mlopt.stage import run_stage

    raw = sys.stdin.read()
    try:
        inp = json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError as e:
        print(f"mlopt stage: stdin is not one JSON value: {e}", file=sys.stderr)
        return 2
    return run_stage(args.name, inp)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlopt", description="pipe models through the verified optimizer"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("models", help="list registered model adapters").set_defaults(fn=_cmd_models)

    doc = sub.add_parser("doctor", help="lift-only diagnostics for a model ref")
    doc.add_argument("model", help="model ref, e.g. whisper:openai/whisper-turbo")
    doc.set_defaults(fn=_cmd_doctor)

    opt = sub.add_parser("optimize", help="optimize a model reference")
    opt.add_argument("model", help="model ref, e.g. whisper:openai/whisper-turbo")
    opt.add_argument(
        "--backend",
        default="torch",
        # Only torch resolves today — onnx/host-tensor sinks exist but
        # have no Source/Composer wiring here yet; offering them would
        # be a lie (review finding).
        choices=["torch"],
    )
    opt.add_argument("--rules", default=None, help="rule set (e.g. bf16_safe, DEFAULT)")
    opt.add_argument("--iters", type=int, default=4)
    opt.add_argument(
        "--adapter-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="adapter load() option, repeatable (e.g. "
        "--adapter-opt arm=f0n --adapter-opt device=cpu; "
        "kokoro: arm/decoder|decoder-nc|f0n|textenc|textenc-p|full, device, "
        "dtype, voice, kokoro_src; values coerce bool/int/float)",
    )
    opt.add_argument("--no-verify", action="store_true")
    opt.add_argument(
        "--deliver",
        action="store_true",
        help="deliver the fast artifact (region_fuse/cuda_graph runner) "
        "and report measured eager-vs-delivered timings",
    )
    opt.add_argument(
        "--bench-iters",
        type=int,
        default=50,
        help="timed forwards per side for --deliver (default 50)",
    )
    opt.add_argument(
        "--deliver-budget",
        type=int,
        default=0,
        metavar="N",
        help="run the delivery GAME with N moves (runner configs are "
        "measured under the verify gate; requires gamekit — falls "
        "back to the heuristic with a note when absent)",
    )
    opt.add_argument(
        "--deliver-player",
        default="playbook",
        help="registry player for the delivery game "
        "(random|playbook|linear|linear-greedy|linear-cold; "
        "default playbook)",
    )
    opt.add_argument(
        "--deliver-heuristic",
        action="store_true",
        help="force the deterministic heuristic runner selection "
        "even when --deliver-budget is given",
    )
    opt.set_defaults(fn=_cmd_optimize)

    exp = sub.add_parser("export", help="export the optimized term as a .hnl program")
    exp.add_argument("model", help="model ref, e.g. whisper:openai/whisper-turbo")
    exp.add_argument("--backend", default="torch", choices=["torch"])
    exp.add_argument("--rules", default=None, help="rule set (e.g. bf16_safe, DEFAULT)")
    exp.add_argument("--iters", type=int, default=4)
    exp.add_argument(
        "--adapter-opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="adapter load() option, repeatable (same as optimize)",
    )
    exp.add_argument("--out", default=None, help="output .hnl path (default <ref>.hnl)")
    exp.add_argument(
        "--sidecar",
        action="store_true",
        help="write <out>.safetensors beside the artifact and emit "
        "tensor.load() for Param leaves above the inline threshold "
        "(requires an hnl-run built with tensor.load to evaluate)",
    )
    exp.add_argument(
        "--sidecar-dtype",
        choices=["F64", "F32"],
        default="F64",
        help="safetensors payload dtype for --sidecar (default F64 — "
        "exact; F32 halves the binary for f32-native weights)",
    )
    exp.add_argument(
        "--check",
        action="store_true",
        help="run the artifact through hnl-run when the binary is available",
    )
    exp.add_argument(
        "--verify-run",
        action="store_true",
        help=(
            "numeric gate: evaluate the emitted program through hnl-run "
            "on the spec's probe inputs and compare against the torch "
            "reference (single-tensor outputs only; declines loudly "
            "otherwise)"
        ),
    )
    exp.set_defaults(fn=_cmd_export)

    stg = sub.add_parser(
        "stage",
        help="meta-tier pipeline stage — one JSON value on stdin, "
        "one JSON record on stdout (the kit-mlopt protocol)",
    )
    stg.add_argument("name", choices=["lift", "search", "verify", "deliver"])
    stg.set_defaults(fn=_cmd_stage)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
