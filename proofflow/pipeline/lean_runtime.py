from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Tuple

from ..lean_check import LeanServer
from .specs import LeanSpec


@dataclass
class LeanRuntimeMetrics:
    compile_seconds: float = 0.0
    compile_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_seconds": round(self.compile_seconds, 6),
            "node_count": self.compile_count,
            "avg_seconds_per_node": round(
                self.compile_seconds / self.compile_count,
                6,
            )
            if self.compile_count
            else None,
        }


class LeanRuntime:
    """Small wrapper that centralizes Lean lifetime and metrics."""

    def __init__(self, spec: LeanSpec, *, lean_server: LeanServer | None = None) -> None:
        self.spec = spec
        self.owned = lean_server is None
        self.server = lean_server or LeanServer(
            project_path=spec.mathlib_path,
            backend=spec.backend,
            pool_size=spec.pool_size,
            temp_root=str(spec.temp_dir) if spec.temp_dir is not None else None,
        )
        self.metrics = LeanRuntimeMetrics()
        self._ready = False

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        if self.spec.backend == "persistent_lsp":
            await self.server.ensure_ready()
        self._ready = True

    async def check(self, lean_code: str, *, job_id: str) -> Tuple[bool, bool, Any]:
        started = time.perf_counter()
        result = await self.server.check_lean_string_async(
            lean_code,
            temp_root=str(self.spec.temp_dir) if self.spec.temp_dir is not None else None,
            job_id=job_id,
        )
        self.metrics.compile_seconds += time.perf_counter() - started
        self.metrics.compile_count += 1
        return result

    async def aclose(self) -> None:
        if self.owned:
            await self.server.aclose()
        self._ready = False
