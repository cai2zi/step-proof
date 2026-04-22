"""Load and render system/user prompt templates under prompts/{proof,calc}/."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal

TaskProfile = Literal["proof", "calc"]

_PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"


def prompts_root() -> Path:
    return _PROMPTS_ROOT


def _read(rel: str) -> str:
    path = _PROMPTS_ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def render_template(template: str, **kwargs: Any) -> str:
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template placeholder: {e}") from e


def build_chat_messages(
    profile: TaskProfile,
    stage: Literal[
        "dag",
        "formalize_claim",
        "formalize_context",
        "prove",
        "prove_negation",
    ],
    **kwargs: Any,
) -> List[Dict[str, str]]:
    """Return [system, user] messages for the given profile and stage."""
    if stage == "dag":
        system = _read(f"{profile}/system/dag.md")
        user = render_template(_read(f"{profile}/user/dag.md"), **kwargs)
    elif stage == "formalize_claim":
        system = _read(f"{profile}/system/formalize.md")
        user = render_template(_read(f"{profile}/user/formalize_claim.md"), **kwargs)
    elif stage == "formalize_context":
        system = _read(f"{profile}/system/formalize.md")
        user = render_template(_read(f"{profile}/user/formalize_context.md"), **kwargs)
    elif stage == "prove":
        system = _read(f"{profile}/system/prove.md")
        user = render_template(_read(f"{profile}/user/prove.md"), **kwargs)
    elif stage == "prove_negation":
        system = _read(f"{profile}/system/prove.md")
        user = render_template(_read(f"{profile}/user/prove_negation.md"), **kwargs)
    else:
        raise ValueError(f"Unknown prompt stage: {stage}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
