from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

from omegaconf import OmegaConf


def _prepend_path_list(existing: str, paths: List[Path]) -> str:
    resolved = [str(path.resolve()) for path in paths if path.exists()]
    parts = resolved + [part for part in existing.split(os.pathsep) if part]
    deduped: List[str] = []
    seen = set()
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return os.pathsep.join(deduped)


def _infer_verl_root(project_root: Path) -> Path:
    env_root = os.environ.get("VERL_ROOT")
    if env_root:
        return Path(env_root)
    sibling = project_root.parent / "verl"
    if sibling.exists():
        return sibling
    return Path(os.environ.get("CZX_ROOT", "/data/run01/scyb202/czx")) / "verl"


def _python_site_paths(python_bin: str) -> List[Path]:
    script = (
        "import json, site, sysconfig; "
        "paths = []; "
        "paths.extend(site.getsitepackages()); "
        "user = site.getusersitepackages(); "
        "paths.append(user); "
        "paths.append(sysconfig.get_paths().get('purelib', '')); "
        "paths.append(sysconfig.get_paths().get('platlib', '')); "
        "print(json.dumps([p for p in paths if p]))"
    )
    try:
        output = subprocess.check_output([python_bin, "-c", script], text=True)
    except Exception:
        return []
    return [Path(path) for path in json.loads(output)]


def _resolve_config_path(config_path: Path) -> Path:
    if config_path.is_absolute() and config_path.exists():
        return config_path
    cwd_candidate = (Path.cwd() / config_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = (Path(__file__).resolve().parents[2] / config_path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return cwd_candidate


def _config_repo_root(config_path: Path) -> Path:
    # configs/rl/fdg_grpo.yaml -> repo root
    if config_path.parent.name == "rl" and config_path.parent.parent.name == "configs":
        return config_path.parent.parent.parent.resolve()
    return Path(__file__).resolve().parents[2]


def _last_override_value(overrides: List[str], key: str) -> str | None:
    prefix = f"{key}="
    add_prefix = f"+{key}="
    value: str | None = None
    for item in overrides:
        if item.startswith(prefix):
            value = item[len(prefix) :]
        elif item.startswith(add_prefix):
            value = item[len(add_prefix) :]
    return value


def _ray_connects_existing_cluster(env: dict[str, str]) -> bool:
    address = env.get("RAY_ADDRESS") or env.get("RAY_REDIS_ADDRESS")
    return bool(address and address.lower() not in {"", "local", "none"})


def _should_set_ray_num_gpus(env: dict[str, str]) -> bool:
    flag = env.get("RL_SET_RAY_NUM_GPUS", "auto").lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    return not _ray_connects_existing_cluster(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FDG builder GRPO training with verl.")
    parser.add_argument("--config", type=Path, default=Path("configs/rl/fdg_grpo.yaml"))
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides forwarded to verl.")
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    raw_cfg = OmegaConf.load(str(config_path))
    cfg_preview = OmegaConf.to_container(raw_cfg, resolve=True)
    project_root_value = Path(str(cfg_preview.get("project_root") or "."))
    project_root = (
        project_root_value.resolve()
        if project_root_value.is_absolute()
        else (_config_repo_root(config_path) / project_root_value).resolve()
    )
    raw_cfg.project_root = str(project_root)
    cfg = OmegaConf.to_container(raw_cfg, resolve=True)
    python_bin = str(cfg.get("python") or sys.executable)
    verl_entry = str(cfg.get("verl_entry") or "verl.trainer.main_ppo")
    overrides: List[str] = [str(item) for item in list(cfg.get("overrides") or [])]
    overrides.extend(list(args.overrides))

    env = os.environ.copy()
    pythonpath = _prepend_path_list(
        env.get("PYTHONPATH", ""),
        [project_root, _infer_verl_root(project_root), *_python_site_paths(python_bin)],
    )
    env["PYTHONPATH"] = pythonpath
    if not any("ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH" in item for item in overrides):
        overrides.append(f"+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={pythonpath}")
    if (
        _should_set_ray_num_gpus(env)
        and not any("ray_kwargs.ray_init.num_gpus" in item for item in overrides)
    ):
        overrides.append(f"+ray_kwargs.ray_init.num_gpus={os.environ.get('RL_NUM_GPUS', '6')}")
    if not any(item.startswith("hydra.run.dir=") for item in overrides):
        experiment_name = _last_override_value(overrides, "trainer.experiment_name") or "fdg_builder_grpo"
        results_root = Path(os.environ.get("RL_RESULTS_ROOT", project_root / "results"))
        if not results_root.is_absolute():
            results_root = project_root / results_root
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hydra_run_dir = results_root / experiment_name / "hydra" / timestamp
        overrides.append(f"hydra.run.dir={hydra_run_dir}")

    command = [python_bin, "-m", verl_entry] + overrides
    print("Launching:", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_root, env=env, check=True)


if __name__ == "__main__":
    main()
