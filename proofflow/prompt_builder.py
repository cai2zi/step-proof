"""Load and render system/user prompt templates under prompts/."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal

PromptStage = Literal["fdg", "formalize_obligation", "prove"]

_PROMPTS_ROOT = Path(__file__).resolve().parent.parent / "prompts"


def prompts_root() -> Path:
    return _PROMPTS_ROOT


def _read(rel: str) -> str:
    path = _PROMPTS_ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _stage_file(role: str, stage: PromptStage, variant: str) -> str:
    suffix = "" if variant == "default" else f".{variant}"
    return f"{role}/{stage}{suffix}.md"


def render_template(template: str, **kwargs: Any) -> str:
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template placeholder: {e}") from e


def build_chat_messages(
    stage: PromptStage,
    *,
    prompt_variant: str = "default",
    **kwargs: Any,
) -> List[Dict[str, str]]:
    """Return chat messages for the given FDG prompt stage and variant."""
    variant = (prompt_variant or "default").strip()
    system = _read(_stage_file("system", stage, variant))
    user = render_template(_read(_stage_file("user", stage, variant)), **kwargs)
    messages: List[Dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages
