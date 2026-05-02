from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch FDG builder GRPO training with verl.")
    parser.add_argument("--config", type=Path, default=Path("configs/rl/fdg_grpo.yaml"))
    parser.add_argument("overrides", nargs="*", help="Extra Hydra overrides forwarded to verl.")
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(str(args.config)), resolve=True)
    python_bin = str(cfg.get("python") or sys.executable)
    verl_entry = str(cfg.get("verl_entry") or "verl.trainer.main_ppo")
    project_root = Path(str(cfg.get("project_root") or ".")).resolve()
    overrides: List[str] = [str(item) for item in list(cfg.get("overrides") or [])]
    overrides.extend(list(args.overrides))

    command = [python_bin, "-m", verl_entry] + overrides
    print("Launching:", " ".join(command))
    subprocess.run(command, cwd=project_root, check=True)


if __name__ == "__main__":
    main()
