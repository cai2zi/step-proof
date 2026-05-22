# Rollout Rerank Math-Verify Experiment

This experiment tests whether step-proof reranking improves final-answer accuracy.

Pipeline:

1. Generate `n=4` rollouts with 8 single-GPU vLLM workers directly from `bench_path`.
2. Flatten rollouts and run the three FDG step-proof stages.
3. Score each rollout by proved node count, build Math-Verify inputs, evaluate, and summarize random, step-proof, and pass@k accuracy.

Default environments:

- Set `CZX_ROOT` to the machine workspace root, for example `/data/run01/scyb202/czx`.
- Set `LEAN4_PYTHON` to the Python used by rollout, step-proof, and Math-Verify.

## Commands

From `${CZX_ROOT}/step-proof`:

```bash
bash experiments/rollout_rerank_math_verify/scripts/run_all.sh
```

The three stages can also be run independently:

```bash
bash experiments/rollout_rerank_math_verify/scripts/01_rollout.sh base
bash experiments/rollout_rerank_math_verify/scripts/02_step_proof.sh base
bash experiments/rollout_rerank_math_verify/scripts/03_eval.sh base
```

The pipeline config only selects stage configs:

```yaml
rollout_config: base
step_proof_config: base
eval_config: base
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

To evaluate only selected benches, edit `data.sources` in `configs/rollout/base.yaml`:

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

Under `${CZX_ROOT}/czx_work/step-proof/rollout_rerank_math_verify/outputs`:

- `rollouts/rollout_<rollout_name>/rollout_raw.jsonl`
- `rollouts/rollout_<rollout_name>/rollout_flat.parquet`
- `rollouts/rollout_<rollout_name>/manifest.json`
- `step_proofs/step_proof_<step_proof_name>/step_proof_results/result_stage3/stage3_results.jsonl`
- `step_proofs/step_proof_<step_proof_name>/scores.jsonl`
- `step_proofs/step_proof_<step_proof_name>/selected_step_proof.jsonl`
- `step_proofs/step_proof_<step_proof_name>/math_verify/random_seed_*_eval.jsonl`
- `step_proofs/step_proof_<step_proof_name>/math_verify/step_proof_best_eval.jsonl`
- `step_proofs/step_proof_<step_proof_name>/math_verify/all_rollouts_eval.jsonl`
- `step_proofs/step_proof_<step_proof_name>/summary/metrics.json`
