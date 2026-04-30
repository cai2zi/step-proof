import copy
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import tiktoken
from openai import OpenAI


def remove_imports(lean_code: str) -> str:
    lines_to_remove = [
        "import Mathlib",
        "import Aesop",
        "set_option maxHeartbeats 0",
        "open BigOperators Real Nat Topology Rat Filter",
    ]
    code_lines = lean_code.split("\n")
    filtered_lines = []
    for line in code_lines:
        line_stripped = line.strip()
        if line_stripped not in lines_to_remove:
            filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def count_tokens(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += len(encoding.encode(msg["content"]))
    return total


def trim_to_limit(messages: List[Dict[str, str]], max_tokens: int = 1000000000) -> List[Dict[str, str]]:
    while count_tokens(messages) > max_tokens and len(messages) > 4:
        messages.pop(1)
    return messages


def extract_answer_openai(response) -> Tuple[bool, str]:
    try:
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            content = getattr(msg, "content", None)
            if content:
                return True, content
            return False, "Empty content in response."
        return False, "No choices in response."
    except Exception as e:
        return False, f"Failed to parse response: {e}"


class LLMManager:
    """OpenAI-compatible chat API.

    If ``system_prompt_path`` is set, its contents are prepended as a ``system`` message
    when the outgoing ``messages`` list does not already start with ``role: system``.
    For the prompt-template pipeline, pass ``system_prompt_path=None`` and supply
    full ``[system, user, ...]`` messages from ``prompt_builder``.
    """

    def __init__(
        self,
        model_info: Dict[str, Any],
        system_prompt_path: Optional[str] = None,
        **kwargs,
    ):
        self.model_info = model_info
        self.client = None
        self.system_prompt = (
            open(system_prompt_path, "r", encoding="utf-8").read()
            if system_prompt_path
            else None
        )

        if "base_url" not in self.model_info:
            raise RuntimeError(
                "This minimal build only supports remote APIs: model_info must include "
                '"base_url" (OpenAI-compatible).'
            )

        base_url = self.model_info["base_url"]
        self.client = OpenAI(
            base_url=base_url,
            api_key=self.model_info.get("api_key"),
            http_client=httpx.Client(
                base_url=base_url,
                follow_redirects=True,
                verify=False,
                timeout=60 * 20,
            ),
            **kwargs,
        )

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        logs: Optional[list] = None,
        max_new_tokens: int = 16384,
        temperature: float = 0.9,
    ) -> Tuple[str, List[Dict[str, str]]]:
        start_time = time.time()
        success_flag = False
        result_text = ""
        response_payload = None
        model_name = self.model_info["model"]
        messages = copy.deepcopy(messages)
        new_tokens = None

        if self.system_prompt:
            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": self.system_prompt})

        try:
            token_limit = 40000 if "goedel" in model_name.lower() else 300000
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=trim_to_limit(messages, token_limit),
                temperature=temperature,
            )
            success_flag, result_text = extract_answer_openai(completion)
            response_payload = result_text if success_flag else completion
            new_tokens = completion.usage.completion_tokens
        except Exception as e:
            success_flag = False
            result_text = f"Request failed: {e}"
            response_payload = str(e)
            print(f"Error during API call: {e}")

        end_time = time.time()
        messages.append({"role": "assistant", "content": response_payload})

        if logs is not None:
            logs.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time,
                    "model": model_name,
                    "messages": messages,
                    "response": response_payload,
                    "success": success_flag,
                    "base_url": self.model_info.get("base_url"),
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "generated_tokens": new_tokens,
                }
            )

        return result_text, messages
