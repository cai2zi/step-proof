from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ..runtime_common import utc_now_iso, write_json_atomic
from .specs import ExperimentSpec, JsonDict, StageSpec


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_fingerprint(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def stage_fingerprint(stage: StageSpec, *, schema_version: str) -> str:
    return stable_fingerprint(
        {
            "schema_version": schema_version,
            "stage": stage.fingerprint_payload(),
        }
    )


@dataclass(frozen=True)
class StageArtifacts:
    name: str
    root: Path
    results_jsonl: Path
    failed_jsonl: Path
    checkpoint_dir: Path
    metrics_json: Path
    fingerprint: str


class ArtifactManager:
    """Owns experiment artifacts and resume fingerprints."""

    def __init__(self, spec: ExperimentSpec) -> None:
        self.spec = spec
        self.root = spec.artifact.root

    def stage(self, stage: StageSpec) -> StageArtifacts:
        stage_root = self.root / f"result_{stage.name}"
        label = stage.name.replace("stage", "")
        return StageArtifacts(
            name=stage.name,
            root=stage_root,
            results_jsonl=stage_root / f"{stage.name}_results.jsonl",
            failed_jsonl=stage_root / f"{stage.name}_failed.jsonl",
            checkpoint_dir=stage_root / f"{stage.name}_ckpt",
            metrics_json=self.root / "stats" / f"{stage.name}_runtime_stats.json",
            fingerprint=stage_fingerprint(
                stage,
                schema_version=self.spec.artifact.schema_version,
            ),
        )

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "stats").mkdir(parents=True, exist_ok=True)
        for stage in self.spec.stages:
            artifacts = self.stage(stage)
            artifacts.root.mkdir(parents=True, exist_ok=True)
            artifacts.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            artifacts.metrics_json.parent.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, *, extra: Optional[JsonDict] = None) -> Path:
        payload: JsonDict = {
            "schema_version": self.spec.artifact.schema_version,
            "created_at": utc_now_iso(),
            "experiment": self.spec.to_dict(),
            "stage_fingerprints": {
                stage.name: self.stage(stage).fingerprint for stage in self.spec.stages
            },
        }
        if extra:
            payload["extra"] = dict(extra)
        path = self.root / "manifest.json"
        write_json_atomic(path, payload)
        return path

    @staticmethod
    def assert_checkpoint_fingerprint(
        checkpoint: JsonDict,
        *,
        expected_fingerprint: str,
        stage_name: str,
    ) -> None:
        existing = str((checkpoint.get("meta") or {}).get("stage_fingerprint") or "")
        if existing and existing != expected_fingerprint:
            rid = (checkpoint.get("meta") or {}).get("record_id", "<unknown>")
            raise RuntimeError(
                f"Record {rid} checkpoint for {stage_name} was created with "
                f"stage_fingerprint={existing!r}, but current run uses "
                f"{expected_fingerprint!r}. Use a new experiment name or disable resume."
            )

    @staticmethod
    def stamp_meta(meta: Dict[str, Any], *, stage_name: str, fingerprint: str) -> None:
        meta["schema_version"] = "step-proof-v2"
        meta["stage_name"] = stage_name
        meta["stage_fingerprint"] = fingerprint
