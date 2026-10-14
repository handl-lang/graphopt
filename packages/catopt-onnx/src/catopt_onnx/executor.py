"""The lowered ONNX runnable — an ``onnxruntime`` inference session.

``Sink.lower`` materialises an IR program as an :class:`OnnxExecutor`:
the ONNX ``ModelProto`` plus the session that runs it.  It satisfies
``catopt_core.ports.Executor`` (``forward(*xs)``) — the port every
lowered module already conforms to — so the orchestrator's routing,
verify and task-metric machinery reach it without knowing it is ONNX.

The executor is also the *delivery*: :meth:`OnnxExecutor.save` writes
the artifact, which is what an ONNX runtime at the edge consumes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from catopt_onnx.arrays import ort_type_to_numpy, to_numpy

__all__ = ["OnnxExecutor"]

#: The default execution provider list — CPU only, and one thread.
#: The sink is a *verification* runtime first: a bounded, deterministic
#: session keeps a four-way-parallel development host responsive and
#: makes the numeric comparison reproducible.
DEFAULT_PROVIDERS: tuple[str, ...] = ("CPUExecutionProvider",)


class OnnxExecutor:
    """A lowered IR program, executable through ``onnxruntime``.

    Parameters
    ----------
    model : onnx.ModelProto
        The artifact :func:`catopt_onnx.lower.lower_ir` produced.
    providers : tuple[str, ...]
        onnxruntime execution providers, in preference order.
    intra_op_threads : int
        Threads one inference may use.  ``1`` (the default) keeps a
        verify run from fanning out.

    """

    def __init__(
        self,
        model: onnx.ModelProto,
        *,
        providers: tuple[str, ...] = DEFAULT_PROVIDERS,
        intra_op_threads: int = 1,
    ) -> None:
        """Build the inference session over ``model``."""
        self._model = model
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=list(providers),
        )
        self._inputs = list(self._session.get_inputs())
        self._input_dtypes = [
            ort_type_to_numpy(i.type) for i in self._inputs
        ]
        self._output_names = [
            o.name for o in self._session.get_outputs()
        ]

    @property
    def model(self) -> onnx.ModelProto:
        """Return the ONNX artifact this executor runs."""
        return self._model

    @property
    def input_names(self) -> tuple[str, ...]:
        """Return the graph's input names, in call order."""
        return tuple(i.name for i in self._inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return the graph's output names, in call order."""
        return tuple(self._output_names)

    def save(self, path: str | Path) -> Path:
        """Write the artifact to ``path`` and return it.

        This is the delivery step: the file is what a non-torch ONNX
        runtime consumes.
        """
        target = Path(path)
        target.write_bytes(self._model.SerializeToString())
        return target

    def forward(self, *xs: Any) -> Any:
        """Run the graph on positional inputs ``xs``.

        Inputs are converted to NumPy and cast to the dtype the graph
        declares for that input (see :mod:`catopt_onnx.arrays` for the
        dtype policy).  Returns a single array for a single-output
        graph, a tuple of arrays otherwise.
        """
        if len(xs) != len(self._inputs):
            raise ValueError(
                f"{self._model.graph.name}: expected "
                f"{len(self._inputs)} input(s) "
                f"({', '.join(i.name for i in self._inputs)}), "
                f"got {len(xs)}"
            )
        feeds = {
            info.name: np.ascontiguousarray(
                to_numpy(value), dtype=dtype
            )
            for info, dtype, value in zip(
                self._inputs, self._input_dtypes, xs, strict=True
            )
        }
        outs = self._session.run(None, feeds)
        return outs[0] if len(outs) == 1 else tuple(outs)
