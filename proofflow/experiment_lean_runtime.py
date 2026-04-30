from __future__ import annotations

from dataclasses import dataclass

from .lean_check import LeanServer


@dataclass(frozen=True)
class LeanRuntimeConfig:
    mathlib_path: str
    lean_backend: str
    lean_check_concurrency: int
    lean_worker_pool_size: int = 0
    lean_temp_dir: str | None = None
    
    @property
    def pool_size(self) -> int:
        if self.lean_worker_pool_size > 0:
            return int(self.lean_worker_pool_size)
        return max(1, int(self.lean_check_concurrency))


class ExperimentLeanRuntime:
    def __init__(self, config: LeanRuntimeConfig) -> None:
        self.config = config
        self.lean_server = LeanServer(
            project_path=self.config.mathlib_path,
            backend=self.config.lean_backend,
            pool_size=self.config.pool_size,
            temp_root=self.config.lean_temp_dir,
        )
        self._ready = False

    async def ensure_ready(self) -> None:
        if self._ready:
            return
        print(
            "[lean-runtime] initializing shared Lean runtime",
            f"(backend={self.config.lean_backend}, pool_size={self.config.pool_size}) ...",
        )
        if self.config.lean_backend == "persistent_lsp":
            await self.lean_server.ensure_ready()
            print("[lean-runtime] persistent LSP workers are ready.\n")
        else:
            print(
                "[lean-runtime] backend does not support eager worker prewarm; "
                "continuing without persistent pool warmup.\n"
            )
        self._ready = True

    async def aclose(self) -> None:
        await self.lean_server.aclose()
        self._ready = False
