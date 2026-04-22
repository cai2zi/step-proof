from __future__ import annotations

from typing import Dict, List, Optional


class LocalLLMManager:
    """Wrap vllm.LLM for in-process batch generation with chat-template support."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 2,
        max_tokens: int = 8192,
        temperature: float = 0.2,
        token_limit: int = 32768,
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.9,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.token_limit = token_limit
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_model_len=token_limit,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def _to_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _token_count(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def batch_generate(
        self,
        message_batches: List[List[Dict[str, str]]],
    ) -> List[Optional[str]]:
        prompts = [self._to_prompt(messages) for messages in message_batches]
        results: List[Optional[str]] = [None] * len(prompts)
        valid_indices: List[int] = []
        valid_prompts: List[str] = []

        for idx, prompt in enumerate(prompts):
            if self._token_count(prompt) > self.token_limit:
                continue
            valid_indices.append(idx)
            valid_prompts.append(prompt)

        if valid_prompts:
            outputs = self.llm.generate(valid_prompts, self.sampling_params)
            for out_idx, prompt_idx in enumerate(valid_indices):
                results[prompt_idx] = outputs[out_idx].outputs[0].text
        return results
