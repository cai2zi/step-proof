from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from proofflow.fdg_graph import build_proof_obligation_from_fact
from proofflow.lean_check import LeanServer

from .formalizer_bridge import FormalizerBridge
from .parser import parse_fdg_candidate
from .prover_bridge import ProverBridge
from .reward_components import (
    count_truthy,
    score_final_answers,
    score_formalizer_pass,
    score_prover_pass,
    score_structure,
)
from .reward_types import (
    BridgeFactTask,
    CandidateGraphInput,
    FDGRLEvaluatorConfig,
    FactRewardTrace,
    GraphRewardBreakdown,
    LeanRuntimeConfig,
    ModelRuntimeConfig,
    RewardWeights,
)


def load_evaluator_config(config_path: str | Path) -> FDGRLEvaluatorConfig:
    cfg = OmegaConf.to_container(OmegaConf.load(str(config_path)), resolve=True)
    weights = RewardWeights(**dict(cfg.get("weights") or {}))
    runtime = dict(cfg.get("runtime") or {})
    formalizer = ModelRuntimeConfig(**dict(runtime.get("formalizer") or {}))
    prover = ModelRuntimeConfig(**dict(runtime.get("prover") or {}))
    lean = LeanRuntimeConfig(**dict(runtime.get("lean") or {}))
    return FDGRLEvaluatorConfig(
        weights=weights,
        formalizer=formalizer,
        prover=prover,
        lean=lean,
        include_prover=bool(runtime.get("include_prover", True)),
    )


class FDGRLEvaluator:
    def __init__(
        self,
        config: FDGRLEvaluatorConfig,
        *,
        formalizer_bridge: Optional[FormalizerBridge] = None,
        prover_bridge: Optional[ProverBridge] = None,
        lean_server: Optional[LeanServer] = None,
        owned_lean_server: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.lean_server = lean_server
        self.owned_lean_server = (
            owned_lean_server if owned_lean_server is not None else lean_server is None
        )
        self.formalizer_bridge = formalizer_bridge
        self.prover_bridge = prover_bridge
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    async def _ensure_runtime(self) -> None:
        need_lean_server = self.formalizer_bridge is None or (
            self.config.include_prover and self.prover_bridge is None
        )
        if need_lean_server and self.lean_server is None:
            pool_size = self.config.lean.worker_pool_size or self.config.lean.check_concurrency
            self.lean_server = LeanServer(
                project_path=self.config.lean.mathlib_path,
                backend=self.config.lean.backend,
                pool_size=pool_size,
                temp_root=self.config.lean.temp_dir,
            )
        if self.formalizer_bridge is None:
            self.formalizer_bridge = FormalizerBridge(
                self.config.formalizer,
                lean_config=self.config.lean,
                lean_server=self.lean_server,
                owned_lean_server=False,
            )
        await self.formalizer_bridge.start()
        if self.config.include_prover:
            if self.prover_bridge is None:
                self.prover_bridge = ProverBridge(
                    self.config.prover,
                    lean_config=self.config.lean,
                    lean_server=self.lean_server,
                    owned_lean_server=False,
                )
            await self.prover_bridge.start()

    async def aclose(self) -> None:
        if self.formalizer_bridge is not None:
            await self.formalizer_bridge.aclose()
            self.formalizer_bridge = None
        if self.prover_bridge is not None:
            await self.prover_bridge.aclose()
            self.prover_bridge = None
        if self.owned_lean_server and self.lean_server is not None:
            await self.lean_server.aclose()
            self.lean_server = None

    def close(self) -> None:
        loop = self._get_loop()
        loop.run_until_complete(self.aclose())
        loop.close()
        self._loop = None

    async def evaluate_batch(self, inputs: List[CandidateGraphInput]) -> List[GraphRewardBreakdown]:
        if not inputs:
            return []

        parsed_items = [parse_fdg_candidate(item.model_output) for item in inputs]
        breakdowns: List[GraphRewardBreakdown] = []
        valid_payloads: List[tuple[int, Any, List[FactRewardTrace]]] = []

        for index, (candidate, parsed) in enumerate(zip(inputs, parsed_items)):
            errors = list(parsed.report.get("errors") or [])
            warnings = list(parsed.report.get("warnings") or [])
            raw_facts = list((parsed.raw_payload or {}).get("facts") or [])
            num_facts = len(raw_facts)

            if not parsed.valid_json:
                structure_score, length_penalty = score_structure(
                    valid_json=False,
                    validator_passed=False,
                    warning_count=0,
                    num_facts=0,
                    weights=self.config.weights,
                )
                breakdowns.append(
                    GraphRewardBreakdown(
                        record_id=candidate.record_id,
                        score=structure_score,
                        structure_score=structure_score,
                        formalizer_score=0.0,
                        prover_score=0.0,
                        final_answer_score=0.0,
                        length_penalty=length_penalty,
                        valid_json=False,
                        validator_passed=False,
                        num_facts=0,
                        num_non_root_facts=0,
                        num_final_facts=0,
                        num_warnings=0,
                        num_formalized=0,
                        num_proved=0,
                        num_final_verified=0,
                        errors=errors,
                        warnings=warnings,
                        facts=[],
                        parse_error=parsed.parse_error or "",
                    )
                )
                continue

            structure_score, length_penalty = score_structure(
                valid_json=True,
                validator_passed=parsed.validator_passed,
                warning_count=len(warnings),
                num_facts=num_facts,
                weights=self.config.weights,
            )
            if not parsed.validator_passed or parsed.document is None:
                breakdowns.append(
                    GraphRewardBreakdown(
                        record_id=candidate.record_id,
                        score=structure_score - length_penalty,
                        structure_score=structure_score,
                        formalizer_score=0.0,
                        prover_score=0.0,
                        final_answer_score=0.0,
                        length_penalty=length_penalty,
                        valid_json=True,
                        validator_passed=False,
                        num_facts=num_facts,
                        num_non_root_facts=0,
                        num_final_facts=0,
                        num_warnings=len(warnings),
                        num_formalized=0,
                        num_proved=0,
                        num_final_verified=0,
                        errors=errors,
                        warnings=warnings,
                        facts=[],
                    )
                )
                continue

            traces: List[FactRewardTrace] = []
            for fact in parsed.document.facts:
                traces.append(
                    FactRewardTrace(
                        fact_id=fact.fact_id,
                        text=fact.text,
                        parent_fact_ids=list(fact.parent_fact_ids),
                        is_final_answer=bool(fact.is_final_answer),
                        origin=fact.origin,
                        proof_obligation=(
                            {}
                            if not fact.parent_fact_ids
                            else build_proof_obligation_from_fact(parsed.document, fact.fact_id)
                        ),
                    )
                )

            valid_payloads.append((index, parsed.document, traces))
            breakdowns.append(
                GraphRewardBreakdown(
                    record_id=candidate.record_id,
                    score=0.0,
                    structure_score=structure_score,
                    formalizer_score=0.0,
                    prover_score=0.0,
                    final_answer_score=0.0,
                    length_penalty=length_penalty,
                    valid_json=True,
                    validator_passed=True,
                    num_facts=len(parsed.document.facts),
                    num_non_root_facts=0,
                    num_final_facts=0,
                    num_warnings=len(warnings),
                    num_formalized=0,
                    num_proved=0,
                    num_final_verified=0,
                    errors=errors,
                    warnings=warnings,
                    facts=traces,
                )
            )

        if not valid_payloads:
            return breakdowns

        await self._ensure_runtime()
        assert self.formalizer_bridge is not None

        form_tasks: List[BridgeFactTask] = []
        fact_lookup: Dict[tuple[int, str], FactRewardTrace] = {}
        for breakdown_index, document, traces in valid_payloads:
            for fact, trace in zip(document.facts, traces):
                fact_lookup[(breakdown_index, fact.fact_id)] = trace
                if fact.parent_fact_ids:
                    fact_state = fact.model_dump()
                    fact_state["proof_obligation"] = trace.proof_obligation
                    form_tasks.append(BridgeFactTask(sample_index=breakdown_index, fact=fact_state))

        form_results = await self.formalizer_bridge.batch_formalize(form_tasks)
        for result in form_results:
            fact_lookup[(result.sample_index, result.fact_id)].formalizer = result

        if self.config.include_prover and self.prover_bridge is not None:
            prove_tasks: List[BridgeFactTask] = []
            for result in form_results:
                if not result.success:
                    continue
                trace = fact_lookup[(result.sample_index, result.fact_id)]
                fact_state = {
                    "fact_id": trace.fact_id,
                    "text": trace.text,
                    "parent_fact_ids": list(trace.parent_fact_ids),
                    "is_final_answer": trace.is_final_answer,
                    "origin": trace.origin,
                    "proof_obligation": dict(trace.proof_obligation),
                    "formalization": {"lean_code": result.lean_code, "lean_pass": True},
                }
                prove_tasks.append(BridgeFactTask(sample_index=result.sample_index, fact=fact_state))
            prove_results = await self.prover_bridge.batch_prove(prove_tasks)
            for result in prove_results:
                fact_lookup[(result.sample_index, result.fact_id)].prover = result

        for breakdown in breakdowns:
            if not breakdown.validator_passed:
                continue
            non_root_facts = [fact for fact in breakdown.facts if fact.parent_fact_ids]
            final_facts = [fact for fact in breakdown.facts if fact.is_final_answer]
            num_non_root = len(non_root_facts)
            num_final = len(final_facts)
            num_formalized = count_truthy(
                fact.formalizer is not None and fact.formalizer.success for fact in non_root_facts
            )
            num_proved = count_truthy(
                fact.prover is not None and fact.prover.verified for fact in non_root_facts
            )
            num_final_verified = count_truthy(
                fact.prover is not None and fact.prover.verified for fact in final_facts
            )

            formalizer_score = score_formalizer_pass(
                num_formalized=num_formalized,
                num_non_root_facts=num_non_root,
                weights=self.config.weights,
            )
            prover_score = score_prover_pass(
                num_proved=num_proved,
                num_non_root_facts=num_non_root,
                weights=self.config.weights,
            )
            final_score = score_final_answers(
                num_final_verified=num_final_verified,
                num_final_facts=num_final,
                weights=self.config.weights,
            )
            breakdown.formalizer_score = formalizer_score
            breakdown.prover_score = prover_score
            breakdown.final_answer_score = final_score
            breakdown.num_non_root_facts = num_non_root
            breakdown.num_final_facts = num_final
            breakdown.num_formalized = num_formalized
            breakdown.num_proved = num_proved
            breakdown.num_final_verified = num_final_verified
            breakdown.score = (
                breakdown.structure_score
                + formalizer_score
                + prover_score
                + final_score
                - breakdown.length_penalty
            )

        return breakdowns

    def evaluate_batch_sync(self, inputs: List[CandidateGraphInput]) -> List[GraphRewardBreakdown]:
        loop = self._get_loop()
        return loop.run_until_complete(self.evaluate_batch(inputs))
