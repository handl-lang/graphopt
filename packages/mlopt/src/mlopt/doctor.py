"""``mlopt doctor`` — lift-only diagnostics for a model ref.

Prints an honest gap report: whether an adapter claims the ref, its
availability, and — for available adapters — what ``load`` yields
(module type, input count, opaque-leaf count when cheaply countable)
or what ``probe`` reports.  Never downloads weights: heavy adapters
implement ``probe(ref)``; ``load`` is only invoked for available
adapters that don't.
"""

from __future__ import annotations

from typing import Any

from mlopt.models import ModelAdapter, resolve_ref


def _find_adapter(ref: str) -> ModelAdapter | None:
    """Resolve *ref* with the same semantics as ``optimize``/``export``.

    ``resolve_ref`` covers every case: ``name:`` prefix routing, bare
    adapter names, and ``claims`` for other bare refs — one semantics
    across doctor/optimize/export/train_delivery_prior.
    """
    return resolve_ref(ref)


def _probe(adapter: ModelAdapter, ref: str) -> dict[str, Any] | None:
    probe = getattr(adapter, "probe", None)
    if not callable(probe):
        return None
    try:
        info = probe(ref)
    except Exception as e:  # noqa: BLE001 — report, don't hide
        return {"probe_error": f"{type(e).__name__}: {e}"}
    return info if isinstance(info, dict) else None


def _opaque_leaf_count(module: Any) -> int | None:
    """Count parameters/leaves if the module exposes them cheaply."""
    params = getattr(module, "parameters", None)
    if not callable(params):
        return None
    try:
        return sum(1 for _ in params())
    except Exception:  # noqa: BLE001
        return None


def _describe_load(adapter: ModelAdapter, ref: str) -> list[str]:
    try:
        spec = adapter.load(ref)
    except Exception as e:  # noqa: BLE001 — declines are the report
        return [f"load:      declined — {type(e).__name__}: {e}"]
    lines = [
        f"load:      ok — module={type(spec.module).__name__}, inputs={len(spec.inputs or ())}"
    ]
    leaves = _opaque_leaf_count(spec.module)
    if leaves is not None:
        lines[-1] += f", opaque_leaves={leaves}"
    if spec.meta:
        lines.append(f"meta:      keys={sorted(spec.meta)}")
    return lines


def diagnose(ref: str) -> str:
    """Return the doctor report for ``ref`` (imports the catalog)."""
    import mlopt.models.catalog  # noqa: F401 — triggers adapter discovery

    lines = [f"ref:       {ref}"]
    adapter = _find_adapter(ref)
    if adapter is None:
        lines.append("adapter:   none — no registered adapter claims this ref")
        lines.append("hint:      run `mlopt models` to see the catalog")
        return "\n".join(lines)

    lines.append(f"adapter:   {adapter.name}")
    lines.append(f"desc:      {adapter.description}")
    if not adapter.available:
        lines.append(f"status:    unavailable — {adapter.unavailable_reason}")
        info = _probe(adapter, ref)
        if info:
            reason = info.get("reason") or info.get("unavailable_reason")
            if reason:
                lines.append(f"probe:     {reason}")
        lines.append("hint:      run `mlopt models` for the full catalog")
        return "\n".join(lines)

    lines.append("status:    available")
    info = _probe(adapter, ref)
    if info is not None:
        for key, value in sorted(info.items()):
            lines.append(f"probe.{key}: {value}")
        return "\n".join(lines)
    lines.extend(_describe_load(adapter, ref))
    return "\n".join(lines)
