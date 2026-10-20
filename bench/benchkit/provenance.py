"""The uniform provenance contract for emitted artifacts (plan 0032
phase 2b).

Every machine-readable artifact the bench lanes emit — scorecard,
leaderboard, limits table, matrix, the pinnable table records — must
carry the context that produced it: **date** (``generated_utc``), **git
sha/dirty** and **device** (inside ``env`` via
:func:`~bench.benchkit.env.collect_env`), and a **seed** slot.

The contract is enforced, not conventional: emitters build their
payload through :func:`with_provenance`, which stamps the fields and
then runs :func:`require_provenance` — a document lacking any of them
is refused before it is ever written.  ``seed`` is a declared slot
rather than a required *value*: genuinely unseeded lanes record
``"seed": null`` honestly — but the key must be present, so "no seed
concept" is a stated property of the artifact, not an omission.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from bench.benchkit.env import collect_env

#: The ``env`` fields every emitted artifact must carry.
REQUIRED_ENV_FIELDS = (
    "timestamp_utc",
    "git_sha",
    "git_dirty",
    "device",
)

#: Sentinel for :func:`with_provenance`'s ``seed`` — distinguishes an
#: explicit ``None`` ("this lane is unseeded") from a forgotten call.
_UNSET: Any = object()


class ProvenanceError(ValueError):
    """An emitted document that lacks a required provenance field."""


def require_provenance(
    doc: Mapping[str, Any], *, where: str = "artifact"
) -> None:
    """Refuse *doc* when it lacks a required provenance field.

    Raises :class:`ProvenanceError` naming every missing field:
    ``generated_utc``, ``seed``, and each of
    :data:`REQUIRED_ENV_FIELDS` under ``env``.
    """
    missing: list[str] = []
    if not doc.get("generated_utc"):
        missing.append("generated_utc")
    env = doc.get("env")
    if not isinstance(env, Mapping):
        missing.append("env")
    else:
        for field in REQUIRED_ENV_FIELDS:
            if field not in env:
                missing.append(f"env.{field}")
    if "seed" not in doc:
        missing.append("seed")
    if missing:
        raise ProvenanceError(
            f"{where} lacks required provenance: " + ", ".join(missing)
        )


def with_provenance(
    payload: Mapping[str, Any],
    *,
    seed: Any = _UNSET,
    device: str | None = "cpu",
    env: dict | None = None,
) -> dict[str, Any]:
    """Stamp *payload* with the provenance contract; return the doc.

    Fills ``env`` (``collect_env(device)`` unless *env* is supplied),
    ``generated_utc`` and ``seed``, then enforces the contract — the
    returned document is guaranteed emittable.  *seed* is keyword-only
    and must be passed: ``None`` is the honest spelling of an unseeded
    lane, but the caller must say so.  ``device=None`` auto-detects
    (``collect_env``'s cuda-when-available default).
    """
    if seed is _UNSET:
        raise TypeError(
            "with_provenance requires seed= (None for an unseeded lane)"
        )
    doc = dict(payload)
    doc["env"] = dict(env) if env is not None else collect_env(device)
    doc.setdefault(
        "generated_utc",
        datetime.now(UTC).isoformat(timespec="seconds"),
    )
    doc["seed"] = seed
    require_provenance(doc)
    return doc
