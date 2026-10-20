"""``KernelBoard`` — the kernel-engineering episode.

One program, the codegen law bank as the move set, and the gauntlet
as referee.  Actions are rule names; firing ``kern_fuse`` marks a
pointwise region for fusion, ``kern_square_strength`` /
``kern_div_recip`` strength-reduce ops, ``kern_fuse_expand``
unschedules (a legal move — and a losing one on a marked board, the
signal a player must learn).  Reward is normalized
:func:`~mlopt_kernel.cost.kernel_cost` improvement per step; the
episode ends at the horizon or when patience runs out.

The referee — *the existing gauntlet* — fires on the terminal
extraction: the spelled source and the candidate the search picked
must derive as equal.  Two referees plug in through the
``verifier`` seam: ``mlopt_discovery.verifier.verify_law`` (the
domain layer injects it — a fresh-graph law check), or the default
in-graph certificate replay (``EGraph.certificate`` +
``verify_certificate`` — the same proof machinery, no discovery
dependency, so the package stays substrate-side).

The ``gamekit.engine.Board`` protocol is structural — this module
implements ``observe``/``step`` and a ``KernelReport`` carrying the
``reward``/``terminal``/``cost``/``certificate_ok`` fields the
episode driver reads, without importing the game layer (the
``kernelopt`` package stays substrate-side for the private-repo
extraction the split plans).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mlopt_core.egraph import verify_certificate
from mlopt_core.ir import Op
from mlopt_core.search_env import SearchEnv

from mlopt_kernel.cost import kernel_cost
from mlopt_kernel.laws import KERNEL_RULES

__all__ = [
    "KernelBoard",
    "KernelReport",
    "KernelState",
]


def _ops_of(term: Any) -> frozenset[str]:
    """Every op name reachable in *term* — the legal-move probe."""
    ops: set[str] = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            ops.add(t.op)
            stack.extend(t.args)
    return frozenset(ops)


def _has_const(term: Any) -> bool:
    """Whether *term* carries a ``Const`` leaf — the recip site's tell."""
    from mlopt_core.ir import Const

    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Const):
            return True
        if isinstance(t, Op):
            stack.extend(t.args)
    return False


#: The stop move — ending codegen evaluation is a real play (the
#: oracle-call decision the kernel board is about), offered by the
#: domain's legal enumerator, not by the env's rule table.
STOP = "stop"


def _matching_rules(term: Any, rules: tuple) -> frozenset[str]:
    """Rule names whose checked lhs matches a subterm of *term*.

    The observable a real compiler has: which codegen patterns apply
    to this program.  ``match_pattern`` walks every subterm; the
    rule's ``check`` filters the binding (``kern_div_recip``'s
    divisor must be a numeric ``Const``) — the same gate the e-graph
    applies at fire time, so a rule in ``matches`` is one that can
    actually fire.
    """
    from mlopt_core.meta import match_pattern

    subs: list = []
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            subs.append(t)
            stack.extend(t.args)
    out: set[str] = set()
    for r in rules:
        for s in subs:
            m = match_pattern(r.lhs, s)
            if m is None:
                continue
            if r.check is None or r.check(m):
                out.add(r.name)
                break
    return frozenset(out)


@dataclass(frozen=True)
class KernelState:
    """The board's observable — the move set, progress, terminator.

    ``ops`` is the program's op vocabulary; ``matches`` the rule
    names whose checked lhs occurs in the program (which codegen
    site a board carries — the player's deciding signal);
    ``has_const`` flags a ``Const`` leaf — the recip site's tell.
    """

    actions: tuple[str, ...]
    ops: frozenset[str]
    matches: frozenset[str]
    has_const: bool
    steps: int
    done: bool
    cost: float
    baseline: float


@dataclass(frozen=True)
class KernelReport:
    """The step report — the ``gamekit.engine.Report`` field floor."""

    action: Any
    reward: float
    terminal: bool = False
    cost: float = float("nan")
    certificate_ok: bool | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class KernelBoard:
    """The codegen-law board — ``SearchEnv`` under a kernel referee.

    Parameters
    ----------
    term
        The spelled program (the case's IR root).
    rules
        The move set — :data:`~mlopt_kernel.laws.KERNEL_RULES` by
        default.
    cost_fn
        The extraction objective — :func:`kernel_cost` by default.
    verifier
        The referee, ``(src, dst, rules) -> result`` with a
        ``derivable`` member.  ``None`` selects the in-graph
        certificate replay; the ``kernel`` play domain injects
        ``mlopt_discovery.verifier.verify_law``.
    step_cost
        The per-move evaluation charge — trying a codegen law is a
        compile/measure call, so each fired rule costs *step_cost*
        of normalized reward (``0`` disables; the kernel domain
        plays ``0.02``).  ``stop`` is free: declining an evaluation
        is the whole skill the board measures.
    horizon / patience
        Episode bounds — patience defaults to the bank size (a full
        playbook never stalls mid-line).

    """

    def __init__(
        self,
        term: Any,
        rules: Any = KERNEL_RULES,
        *,
        cost_fn: Callable = kernel_cost,
        verifier: Callable | None = None,
        step_cost: float = 0.0,
        horizon: int = 8,
        patience: int | None = None,
    ) -> None:
        """Bind the program, the bank, the referee, and the bounds."""
        self.term = term
        self.rules = tuple(rules)
        self.env = SearchEnv(
            term,
            self.rules,
            cost_fn=cost_fn,
            horizon=horizon,
            patience=len(self.rules) if patience is None else patience,
        )
        self.verifier = verifier
        self.step_cost = step_cost
        self._ops = _ops_of(term)
        self._matches = _matching_rules(term, self.rules)
        self._has_const = _has_const(term)
        self._steps = 0
        self._done = True  # lazily reset on first observe
        self._baseline = float("inf")
        self._certified: bool | None = None

    @property
    def eg(self) -> Any:
        """The working e-graph — exposed for extraction/delivery."""
        return self.env.eg

    @property
    def root(self) -> int:
        """The program's root e-class."""
        return self.env.root

    def extract_best(self) -> Any:
        """Return the current cheapest program under the kernel cost."""
        if self.env.eg is None:
            return self.term
        return self.env._extract()

    def observe(self) -> KernelState:
        """Return the board view — reset lazily on first look."""
        if self._done and self._steps == 0:
            self.env.reset()
            self._done = False
            self._baseline = self.env.cost
        return KernelState(
            actions=self.env.action_names,
            ops=self._ops,
            matches=self._matches,
            has_const=self._has_const,
            steps=self._steps,
            done=self._done,
            cost=self.env.cost,
            baseline=self._baseline,
        )

    def _certify(self) -> bool | None:
        """Run the referee on the terminal extraction — once."""
        if self._certified is not None:
            return self._certified
        best = self.extract_best()
        if best is None or self.env.eg is None:
            self._certified = None
            return None
        if self.verifier is not None:
            self._certified = bool(
                self.verifier(self.term, best, self.rules).derivable
            )
            return self._certified
        try:
            cert = self.env.eg.certificate(self.term, best)
            verify_certificate(self.term, cert)
            self._certified = True
        except Exception:
            self._certified = False
        return self._certified

    def step(self, action: Any) -> tuple[KernelState, KernelReport]:
        """Fire one rule; report cost improvement and the cert verdict.

        ``stop`` ends the episode at no charge — the certified
        extraction the terminal report carries.  Every rule fire
        pays ``step_cost`` off its improvement.
        """
        self.observe()  # ensure reset
        name = (
            action if isinstance(action, str) else action.params["rule"]
        )
        self._steps += 1
        if name == STOP:
            self._done = True
            cert = self._certify()
            return self.observe(), KernelReport(
                action,
                reward=0.0,
                terminal=True,
                cost=self.env.cost,
                certificate_ok=cert,
            )
        res = self.env.step(name)
        self._done = res.done
        cert = self._certify() if self._done else None
        return self.observe(), KernelReport(
            action,
            reward=res.reward - self.step_cost,
            terminal=res.done,
            cost=res.cost,
            certificate_ok=cert,
        )
