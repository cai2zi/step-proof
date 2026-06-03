from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


Message = Dict[str, str]
JsonDict = Dict[str, Any]


@dataclass
class NodeTask:
    record_id: str
    fact_id: str
    messages: List[Message]
    attempt_num: int
    raw_messages: List[Message] | None = None


class PipelineStage(ABC):
    """Minimal stage contract used by the v2 pipeline architecture."""

    name: str

    @abstractmethod
    async def run(self) -> None:
        ...


class NodeLLMStage(PipelineStage):
    """Base contract for formalize/prove-style graph node stages."""

    @abstractmethod
    def load_inputs(self) -> None:
        ...

    @abstractmethod
    def build_prompt(self, record: JsonDict, fact: JsonDict) -> List[Message]:
        ...

    @abstractmethod
    async def validate_output(self, task: NodeTask, generation: JsonDict) -> JsonDict:
        ...

    @abstractmethod
    async def apply_result(self, task: NodeTask, generation: JsonDict) -> None:
        ...
