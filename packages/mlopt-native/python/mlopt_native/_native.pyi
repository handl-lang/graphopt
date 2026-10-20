"""Type stub for the compiled ``mlopt_native._native`` extension.

Mirrors the PyO3 binding surface in ``src/lib.rs``.  Kept next to the
package (maturin's ``python-source`` layout ships it alongside
``__init__.py``), so type checkers and IDEs resolve the extension.
"""

from typing import Any

__version__: str

class NativeEGraph:
    """The Rust search core — see ``mlopt_native.engine``."""

    def __init__(self, build_term: Any) -> None: ...
    @property
    def n_enodes(self) -> int: ...
    @property
    def n_classes(self) -> int: ...
    def find(self, eid: int) -> int: ...
    def union(self, a: int, b: int) -> bool: ...
    def add_leaf(self, key: str) -> int: ...
    def add_term(self, flat: list) -> int: ...
    def add_rule(
        self,
        name: str,
        lhs: Any,
        rhs: Any,
        check: Any = None,
        derive: Any = None,
    ) -> None: ...
    def rebuild(self, classes: list[int] | None = None) -> bool: ...
    def run_iteration(
        self,
        rule_names: list[str],
        max_nodes: int = ...,
        rule_budgets: dict[str, int] | None = None,
    ) -> str: ...
    def finish(self) -> None: ...
    def budget_spent(self) -> dict[str, int]: ...
    def run(
        self,
        rule_names: list[str],
        max_iterations: int = ...,
        max_nodes: int = ...,
        rule_budgets: dict[str, int] | None = None,
    ) -> dict[str, Any]: ...
    def classes(self) -> list[tuple[int, list]]: ...
    def rule_fires(self) -> dict[str, int]: ...
    def extract(
        self,
        root: int,
        pricer: Any,
        fuser: Any = None,
        bill_params: bool = ...,
        foldable: Any = None,
        fusion_epsilon: float = ...,
        overrides: Any = None,
        bans: Any = None,
    ) -> tuple[float, int | None, int, list]: ...

def dag_cost(
    flat: list, pricer: Any, bill_params: bool, foldable: Any
) -> float: ...
def contract_greedy(
    tensors: Any, sizes: dict[int, int], picks: list[int] | None = None
) -> tuple[list[tuple[int, int]], float]: ...
def contract_cost_of_order(
    tensors: Any, sizes: dict[int, int], order: Any
) -> float: ...
def contract_pair_cost(
    x: Any, y: Any, sizes: dict[int, int]
) -> float: ...
def contract_merge(tensors: Any, a: int, b: int) -> list[list[int]]: ...
