"""mlopt-cuda — the CUDA-graph delivery mechanism (plan 0007).

:class:`CudaGraphRunner` — the :class:`~mlopt_core.ports.Runner`
that captures a delivered module into a CUDA graph, collapsing a
carrier executor's per-level kernel launches — or a generic
``IRModule``'s per-leaf tape dispatch — into one replayable graph.
It lives in its own package so the CUDA mechanism — the only
piece of the pipeline that is *device*-coupled, not merely
torch-coupled — is separately installable and separately evolvable.

Delivery runners are composable: ``optimize_model``-family entry
points accept ``runner=CudaGraphRunner()`` or a
``ChainedRunner([TorchCompileRunner(), CudaGraphRunner()])`` — the
compiled-wins precedence lives inside this runner itself.
"""

from mlopt_cuda.runners import CudaGraphRunner

__all__ = ["CudaGraphRunner"]
