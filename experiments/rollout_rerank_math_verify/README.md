# Rollout Rerank Math-Verify Experiment

This experiment tests whether step-proof reranking improves final-answer accuracy.

Pipeline:

1. Prepare a bench subset from `bench_all_normalized`.
2. Generate `n=4` rollouts with 8 single-GPU vLLM workers.
3. Flatten rollouts and run the three FDG step-proof stages.
4. Score each rollout by proved node count and select the best per problem.
5. Build random and step-proof Math-Verify JSONLs.
6. Evaluate with Math-Verify and summarize random, step-proof, and pass@k accuracy.

Default environments:

- Rollout and step-proof: `/root/autodl-tmp/env/lean4/bin/python`
- Math-Verify: `/root/autodl-tmp/env/eval/bin/python`

## Commands

From `/root/autodl-tmp/step-proof`:

```bash
CONFIG=experiments/rollout_rerank_math_verify/configs/experiment.yaml

/root/autodl-tmp/env/lean4/bin/python \
  experiments/rollout_rerank_math_verify/scripts/00_prepare_input.py \
  --config "$CONFIG"

bash experiments/rollout_rerank_math_verify/scripts/01_run_rollout.sh "$CONFIG"

/root/autodl-tmp/env/lean4/bin/python \
  experiments/rollout_rerank_math_verify/scripts/02_flatten_rollouts_for_step_proof.py \
  --config "$CONFIG"

bash experiments/rollout_rerank_math_verify/scripts/03_run_step_proof.sh "$CONFIG"

/root/autodl-tmp/env/lean4/bin/python \
  experiments/rollout_rerank_math_verify/scripts/04_score_step_proof.py \
  --config "$CONFIG"

/root/autodl-tmp/env/lean4/bin/python \
  experiments/rollout_rerank_math_verify/scripts/05_build_math_verify_inputs.py \
  --config "$CONFIG"

bash experiments/rollout_rerank_math_verify/scripts/06_run_math_verify.sh "$CONFIG"

/root/autodl-tmp/env/lean4/bin/python \
  experiments/rollout_rerank_math_verify/scripts/07_summarize_results.py \
  --config "$CONFIG"
```

Default config is a smoke run with `limit_per_source: 10`. Set it to `null` for full data.

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

To evaluate only selected benches, edit `data.sources` in `configs/experiment.yaml`:

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

Under `outputs/${exp_name}`:

- `input/bench.parquet`
- `rollout/rollout_raw.jsonl`
- `step_proof/input/rollout_flat.parquet`
- `step_proof_results/fdg/result_stage3/stage3_results.jsonl`
- `step_proof/scores.jsonl`
- `step_proof/selected_step_proof.jsonl`
- `math_verify/random_seed_*_eval.jsonl`
- `math_verify/step_proof_best_eval.jsonl`
- `math_verify/all_rollouts_eval.jsonl`
- `summary/metrics.json`
