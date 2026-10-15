"""Law-discovery engine for catopt.

The meta-level toolkit that finds, verifies and ranks candidate
rewrite laws over the core IR: shape census (:mod:`.census`),
proposal generators (:mod:`.proposal`, :mod:`.shape_proposal`,
:mod:`.grammar`, the form-level proposer :mod:`.forms` and the
vocabulary proposer :mod:`.propose`), oracles (:mod:`.verifier`,
:mod:`.oracle`), the
impact/verdict probes (:mod:`.impact`, :mod:`.pipeline`), the sqlite
evidence store (:mod:`.evidence`), the coherence catalogue
(:mod:`.coherence`, :mod:`.coherence2`), workload intake
(:mod:`.intake`, :mod:`.workload_gen`), and the emit/vocab/meta-game
helpers, and the object constructor (:mod:`.object_synthesis` — the
ADR-0004 "construction operations" that *build* declared objects for
the evidence store to admit).  The game layer: boards
(:mod:`.arena`, :mod:`.meta_arena`), the generic driver
(:mod:`.engine` — one ``Board`` protocol over all of them), the
player/featurizer abstraction (:mod:`.players` — ``LinearPolicy``,
the one ``LearnedPlayer`` and the per-domain ``DomainSpec`` of
composable feature parts), the meta preset (:mod:`.meta_player`),
the reverse handler (:mod:`.training`), the
generated-kernel machinery (:mod:`.genkernel` — minted handlers +
bindings + gen sink — the lowerings the search itself builds),
the handler-grammar minter (:mod:`.handler_synth` — ADR 0007
layer 2: structured loop/fold/contraction handler bodies verified
against the pre-mint subterm),
and the domain registry + CLI (:mod:`.play` —
``python -m catopt_discovery.play --domain meta|joint|search|torch|gen``).
Formerly the ``tools/law_*.py``
scripts; every module keeps its CLI — invoke as
``python -m catopt_discovery.<module>``.

.. data:: REPO_ROOT

   Absolute path of the repository root, derived from this editable
   file location (``packages/catopt-discovery/src/catopt_discovery``
   is four levels below the root).  Used to reach ``bench`` (a
   repo-root package, not an installed distribution).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

#: The environment override for :func:`cache_dir`.
CACHE_DIR_ENV = "CATOPT_CACHE_DIR"


def cache_dir() -> Path:
    """Directory for regenerable local artifacts.

    ``$CATOPT_CACHE_DIR`` wins; else ``$XDG_CACHE_HOME/catopt``; else
    ``~/.cache/catopt`` — the same convention as
    ``catopt_core.profile.profiles_dir()``.  Holds the intake census
    side-files (``intake/``) and the evidence store, none of which
    belong in the tree: they are large, regenerable, and shared across
    worktrees.
    """
    env = os.environ.get(CACHE_DIR_ENV)
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "catopt"


def intake_dir() -> Path:
    """Return the intake census side-files directory.

    Holds ``intake_corpus.json`` and ``intake_tensors.pt``.
    """
    return cache_dir() / "intake"
