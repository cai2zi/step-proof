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
    lean_api_url: str | None = None
    lean_api_key_env: str | None = None
    lean_server_timeout: int = 300
    lean_server_reuse: bool = True
    lean_server_debug: bool = False
    
    @property
    def pool_size(self) -> int:
        if self.lean_worker_pool_size > 0:
            return int(self.lean_worker_pool_size)
        return max(1, int(self.lean_check_concurrency))


class ExperimentLeanRuntime:
    def __init__(self, config: LeanRuntimeConfig) -> None:
        self.config = config
        if self.config.lean_backend == "kimina_server":
            self.lean_server = LeanServer(
                api_url=self.config.lean_api_url,
                backend=self.config.lean_backend,
                api_key_env=self.config.lean_api_key_env,
                server_timeout=self.config.lean_server_timeout,
                server_reuse=self.config.lean_server_reuse,
                server_debug=self.config.lean_server_debug,
            )
        else:
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
        elif self.config.lean_backend == "kimina_server":
            print(
                "[lean-runtime] using kimina server",
                f"url={self.config.lean_api_url or 'http://localhost:8000'} "
                f"client_concurrency={self.config.lean_check_concurrency}.\n",
            )
        else:
            print(
                "[lean-runtime] backend does not support eager worker prewarm; "
                "continuing without persistent pool warmup.\n"
            )
        self._ready = True

    async def aclose(self) -> None:
        await self.lean_server.aclose()
        self._ready = False
