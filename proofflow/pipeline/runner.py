from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from ..runtime_common import write_json_atomic
from .artifacts import ArtifactManager
from .specs import ExperimentSpec
from .stages import PipelineStage


def current_git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


class PipelineRunner:
    """Thin v2 orchestrator for composable stage objects."""

    def __init__(
        self,
        spec: ExperimentSpec,
        *,
        stage_factory: Callable[[str], PipelineStage],
    ) -> None:
        self.spec = spec
        self.artifacts = ArtifactManager(spec)
        self.stage_factory = stage_factory

    async def run(self, stages: Iterable[str] | None = None) -> None:
        selected = list(stages or [stage.name for stage in self.spec.stages])
        self.artifacts.ensure_layout()
        self.artifacts.write_manifest()
        for stage_name in selected:
            stage = self.stage_factory(stage_name)
            await stage.run()

    def write_summary(self, payload: Dict[str, object]) -> None:
        write_json_atomic(self.spec.artifact.root / "pipeline_summary.json", payload)


class StaticStageFactory:
    def __init__(self, stages: List[PipelineStage]) -> None:
        self._stages = {stage.name: stage for stage in stages}

    def __call__(self, name: str) -> PipelineStage:
        try:
            return self._stages[name]
        except KeyError as exc:
            raise KeyError(f"Unknown pipeline stage {name!r}") from exc
