"""ProofFlow package exports.

Keep package import lightweight so visualization and utility scripts do not
eagerly import optional Lean/Kimina dependencies.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["ProofFlow", "LLMManager", "LeanServer"]


def __getattr__(name: str):
    if name == "LeanServer":
        from .lean_check import LeanServer

        return LeanServer
    if name == "ProofFlow":
        from .proofflow import ProofFlow

        return ProofFlow
    if name == "LLMManager":
        from .utils import LLMManager

        return LLMManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
