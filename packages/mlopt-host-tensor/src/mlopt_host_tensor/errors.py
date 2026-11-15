"""Errors for the host-tensor backend."""

from __future__ import annotations


class HostTensorError(RuntimeError):
    """A host-tensor call or codec failure."""


class UnsupportedHostOpError(HostTensorError):
    """The IR program uses an op the host-tensor sink cannot lower.

    Raised loudly at eval (and priced ``+inf`` at extraction via
    ``Capabilities.supported_ops``) — never a silent approximation.
    """
