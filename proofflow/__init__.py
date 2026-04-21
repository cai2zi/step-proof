"""ProofFlow: NL proof → Lean 4 (minimal package for example.py)."""

from .lean_check import LeanServer
from .proofflow import ProofFlow
from .utils import LLMManager

__version__ = "1.0.0"
__all__ = ["ProofFlow", "LLMManager", "LeanServer"]
