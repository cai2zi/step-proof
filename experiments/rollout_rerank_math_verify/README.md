# Rollout Rerank Math-Verify Experiment

This experiment tests whether step-proof reranking improves final-answer accuracy.

Pipeline:

1. Generate `n=4` rollouts with 8 single-GPU vLLM workers directly from `bench_path`.
2. Flatten rollouts and run the three FDG step-proof stages.
3. Score each rollout by proved node count, build Math-Verify inputs, evaluate, and summarize random, step-proof, and pass@k accuracy.

Default environments:

- Rollout and step-proof: `/root/autodl-tmp/env/lean4/bin/python`
- Math-Verify: `/root/autodl-tmp/env/eval/bin/python`

## Commands

From `/root/autodl-tmp/step-proof`:

```bash
bash experiments/rollout_rerank_math_verify/scripts/run_all.sh default
```

The three stages can also be run independently:

```bash
bash experiments/rollout_rerank_math_verify/scripts/01_rollout.sh qwen3_8b
bash experiments/rollout_rerank_math_verify/scripts/02_step_proof.sh fdg_bug
bash experiments/rollout_rerank_math_verify/scripts/03_eval.sh math_verify
```

The pipeline config only selects stage configs:

```yaml
rollout_config: qwen3_8b
step_proof_config: fdg_bug
eval_config: math_verify
```

Rollout uses one vLLM process per GPU by default. Worker startup and scheduling are controlled by:

```yaml
rollout:
  instances: 8
  gpus: 0,1,2,3,4,5,6,7
  tensor_parallel_size: 1
  parallel_startup: true
  startup_stagger_seconds: 0
```

With `parallel_startup: true`, all vLLM worker processes are started first and then waited on for readiness. During inference, batches are scheduled dynamically: when a worker finishes its current micro-batch, it immediately receives the next pending micro-batch.

To evaluate only selected benches, edit `data.sources` in `configs/rollout/qwen3_8b.yaml`:

```yaml
data:
  sources:
    - gsm8k_main_test
    - math_500_test
    - olympiadbench_oe_to_maths_en_comp
```

Set `sources: null` to use all benches. The supported source names are:

```text
aime25_test
aime_2024
brumo_2025
cmimc_2025
gsm8k_main_test
hmmt_feb_2025
math_500_test
olympiadbench_oe_to_maths_en_comp
omni_math_test
```

## Key Outputs

Under `outputs`:

- `rollouts/rollout_qwen3_8b/rollout_raw.jsonl`
- `rollouts/rollout_qwen3_8b/rollout_flat.parquet`
- `rollouts/rollout_qwen3_8b/manifest.json`
- `step_proofs/step_proof_fdg_bug/step_proof_results/result_stage3/stage3_results.jsonl`
- `step_proofs/step_proof_fdg_bug/scores.jsonl`
- `step_proofs/step_proof_fdg_bug/selected_step_proof.jsonl`
- `step_proofs/step_proof_fdg_bug/math_verify/random_seed_*_eval.jsonl`
- `step_proofs/step_proof_fdg_bug/math_verify/step_proof_best_eval.jsonl`
- `step_proofs/step_proof_fdg_bug/math_verify/all_rollouts_eval.jsonl`
- `step_proofs/step_proof_fdg_bug/summary/metrics.json`
