from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from proofflow.pipeline.fdg_stages import FormalizeStage, GraphBuildStage, ProveStage
from proofflow.pipeline.llm_backends import FakeLLMBackend


class FakeLeanRuntime:
    def __init__(self, *, lean_pass: bool = True, lean_verify: bool = True) -> None:
        self.lean_pass = lean_pass
        self.lean_verify = lean_verify
        self.closed = False

    async def ensure_ready(self) -> None:
        return None

    async def check(self, lean_code: str, *, job_id: str):
        return self.lean_pass, self.lean_verify, []

    async def aclose(self) -> None:
        self.closed = True


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stage1_args(tmp_path: Path) -> argparse.Namespace:
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir()
    pd.DataFrame(
        [
            {
                "id": "case-1",
                "question": "Prove True.",
                "response": "True is trivial.",
            }
        ]
    ).to_parquet(parquet_dir / "rollout_flat.parquet")
    return argparse.Namespace(
        parquet_dir=parquet_dir,
        glob="*.parquet",
        id_column="id",
        question_column="question",
        response_column="response",
        limit=-1,
        out=tmp_path / "stage1" / "graphs.jsonl",
        skipped=tmp_path / "stage1" / "skipped.jsonl",
        failed=tmp_path / "stage1" / "failed.jsonl",
        api_pending=tmp_path / "stage1" / "api_pending.jsonl",
        no_resume=True,
        backend="fake",
        model_path="",
        vllm_instances=1,
        parallel_startup=False,
        startup_stagger_seconds=0.0,
        startup_timeout=1,
        tensor_parallel_size=1,
        gpus="",
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_tokens=128,
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        seed=42,
        top_k=20,
        token_limit=1024,
        batch_size=2,
        max_retries=0,
        fdg_prompt="fdg_origin4_reduce",
        validation_checks={
            "dependency_structure": True,
            "origin_rules": True,
            "introduced_without_parents": True,
            "all_facts_reach_answer": True,
        },
        include_think_in_dag=False,
        chat_template_kwargs_json=None,
        api_model="fake",
        api_base_url="",
        api_key_env="",
        api_concurrency=1,
        api_timeout=1.0,
        api_max_retries=0,
        api_retry_sleep=0.0,
        api_input_token_limit=-1,
        api_tokenizer_path="",
    )


def _stage1_graph_row() -> dict:
    return {
        "meta": {"record_id": "case-1", "schema_version": "fdg-v1"},
        "input": {"problem": "Prove True.", "raw_cot": "True is trivial."},
        "graph": {
            "topo_order": ["f0", "f1"],
            "facts": [
                {
                    "fact_id": "f0",
                    "text": "True",
                    "parent_fact_ids": [],
                    "is_final_answer": False,
                    "origin": "given",
                    "skip": 1,
                },
                {
                    "fact_id": "f1",
                    "text": "True",
                    "parent_fact_ids": ["f0"],
                    "is_final_answer": True,
                    "origin": "answer",
                    "skip": 0,
                },
            ],
            "final_fact_ids": ["f1"],
        },
    }


def _stage2_args(tmp_path: Path) -> argparse.Namespace:
    infile = tmp_path / "stage1" / "graphs.jsonl"
    _write_jsonl(infile, [_stage1_graph_row()])
    return argparse.Namespace(
        infile=infile,
        out=tmp_path / "stage2" / "stage2_results.jsonl",
        failed=tmp_path / "stage2" / "stage2_failed.jsonl",
        checkpoint_dir=tmp_path / "stage2" / "stage2_ckpt",
        limit=-1,
        no_resume=True,
        mathlib_path="",
        lean_backend="subprocess",
        lean_check_concurrency=1,
        lean_worker_pool_size=0,
        lean_temp_dir=tmp_path / "lean",
        backend="fake",
        gpus="",
        dtype="float16",
        gpu_memory_utilization=0.9,
        batch_wait_ms=1,
        max_pending_validation_batches=1,
        formalizer_gpus="",
        formalizer_model_path="fake-formalizer",
        formalizer_instances=1,
        formalizer_parallel_startup=False,
        formalizer_startup_stagger_seconds=0.0,
        formalizer_startup_timeout=1,
        formalizer_tensor_parallel_size=1,
        formalizer_max_tokens=128,
        formalizer_token_limit=1024,
        formalizer_temperature=0.0,
        formalizer_top_p=1.0,
        formalizer_presence_penalty=0.0,
        formalizer_frequency_penalty=0.0,
        formalizer_seed=42,
        formalizer_top_k=20,
        formalizer_chat_template_kwargs_json=None,
        formalizer_prompt="formalize_obligation.context_ablation",
        formalizer_context_mode="parent_only",
        formalizer_retries=0,
        form_batch_size=2,
        api_model="fake",
        api_base_url="",
        api_key_env="",
        api_concurrency=1,
        api_timeout=1.0,
        api_max_retries=0,
        api_retry_sleep=0.0,
        api_input_token_limit=-1,
        api_tokenizer_path="",
        metrics_out=None,
    )


def _stage3_args(tmp_path: Path, stage2_row: dict) -> argparse.Namespace:
    infile = tmp_path / "stage2" / "stage2_results.jsonl"
    _write_jsonl(infile, [stage2_row])
    return argparse.Namespace(
        infile=infile,
        out=tmp_path / "stage3" / "stage3_results.jsonl",
        failed=tmp_path / "stage3" / "stage3_failed.jsonl",
        checkpoint_dir=tmp_path / "stage3" / "stage3_ckpt",
        limit=-1,
        no_resume=True,
        mathlib_path="",
        lean_backend="subprocess",
        lean_check_concurrency=1,
        lean_worker_pool_size=0,
        lean_temp_dir=tmp_path / "lean",
        gpus="",
        dtype="float16",
        gpu_memory_utilization=0.9,
        batch_wait_ms=1,
        max_pending_validation_batches=1,
        prover_gpus="",
        prover_model_path="fake-prover",
        prover_instances=1,
        prover_parallel_startup=False,
        prover_startup_stagger_seconds=0.0,
        prover_startup_timeout=1,
        prover_tensor_parallel_size=1,
        prover_max_tokens=128,
        prover_token_limit=1024,
        prover_temperature=0.0,
        prover_top_p=1.0,
        prover_presence_penalty=0.0,
        prover_frequency_penalty=0.0,
        prover_seed=42,
        prover_top_k=20,
        prover_chat_template_kwargs_json=None,
        prover_prompt="prove.paper_goedel_v2",
        prover_retries=0,
        prove_batch_size=2,
        metrics_out=None,
    )


def test_graph_build_stage_fake_backend(tmp_path):
    fdg_json = {
        "problem_id": "case-1",
        "problem_text": "Prove True.",
        "facts": [
            {
                "fact_id": "f0",
                "text": "True",
                "parent_fact_ids": [],
                "is_final_answer": False,
                "origin": "given",
            },
            {
                "fact_id": "f1",
                "text": "True",
                "parent_fact_ids": ["f0"],
                "is_final_answer": True,
                "origin": "answer",
            },
        ],
    }
    backend = FakeLLMBackend("```json\n" + json.dumps(fdg_json) + "\n```")
    args = _stage1_args(tmp_path)

    asyncio.run(GraphBuildStage(args, backend=backend).run())

    rows = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["meta"]["schema_version"] == "step-proof-v2"
    assert rows[0]["graph"]["final_fact_ids"] == ["f1"]


def test_formalize_stage_fake_backend_and_lean(tmp_path):
    args = _stage2_args(tmp_path)
    backend = FakeLLMBackend("```lean4\ntheorem test : True := by trivial\n```")
    lean = FakeLeanRuntime(lean_pass=True, lean_verify=False)

    asyncio.run(FormalizeStage(args, backend=backend, lean_runtime=lean).run())

    rows = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines()]
    fact = rows[0]["results"]["facts"][1]
    assert fact["form_status"] == "success"
    assert fact["formalization"]["lean_pass"] is True


def test_prove_stage_fake_backend_and_lean(tmp_path):
    stage2_row = _stage1_graph_row()
    stage2_row["results"] = {
        "facts": [
            {
                **stage2_row["graph"]["facts"][0],
                "proof_obligation": {},
                "form_status": "skipped",
                "formalization": {"lean_code": "", "lean_pass": True, "skipped": True},
            },
            {
                **stage2_row["graph"]["facts"][1],
                "proof_obligation": {
                    "problem_name": "prove_f1",
                    "informal_statement_content": "Given True, prove that True.",
                },
                "form_status": "success",
                "formalization": {
                    "lean_code": "theorem test : True := by trivial",
                    "lean_pass": True,
                },
            },
        ]
    }
    args = _stage3_args(tmp_path, stage2_row)
    backend = FakeLLMBackend("```lean4\ntheorem test : True := by trivial\n```")
    lean = FakeLeanRuntime(lean_pass=True, lean_verify=True)

    asyncio.run(ProveStage(args, backend=backend, lean_runtime=lean).run())

    rows = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines()]
    fact = rows[0]["results"]["facts"][1]
    assert fact["prove_status"] == "success"
    assert fact["solved_lemma"]["lean_verify"] is True
