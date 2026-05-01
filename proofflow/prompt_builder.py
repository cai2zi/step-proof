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


def _normalize_prompt_name(stage: PromptStage, prompt_name: str | None) -> str:
    name = str(prompt_name or "").strip()
    if not name:
        raise ValueError(f"Prompt name is required for stage {stage!r}.")
    if "/" in name or "\\" in name:
        raise ValueError(f"Prompt name must be a file stem, got {name!r}.")
    if name.endswith(".md"):
        name = name[:-3]
    return name


def _prompt_file(role: str, stage: PromptStage, prompt_name: str | None) -> str:
    name = _normalize_prompt_name(stage, prompt_name)
    return f"{role}/{name}.md"


def render_template(template: str, **kwargs: Any) -> str:
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise ValueError(f"Missing template placeholder: {e}") from e


def build_chat_messages(
    stage: PromptStage,
    *,
    prompt_name: str | None = None,
    **kwargs: Any,
) -> List[Dict[str, str]]:
    """Return chat messages from prompts/{system,user}/{prompt_name}.md."""
    name = _normalize_prompt_name(stage, prompt_name or stage)
    system = _read(_prompt_file("system", stage, name))
    user = render_template(_read(_prompt_file("user", stage, name)), **kwargs)
    messages: List[Dict[str, str]] = []
    if system.strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages
