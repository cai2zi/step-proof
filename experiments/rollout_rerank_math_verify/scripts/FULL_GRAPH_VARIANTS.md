# Full Graph Variant Scripts

The old bit-numbered `run_full_*.sh` scripts have been replaced by explicit
variant names.

| Old script | New script |
| --- | --- |
| `run_full_000.sh` | `run_full_graph__s1_vllm__f_8b_vllm.sh` |
| `run_full_010.sh` | `run_full_graph__s1_vllm__f_32b_vllm.sh` |
| `run_full_020.sh` | `run_full_graph__s1_vllm__f_8b_api.sh` |
| `run_full_100.sh` | `run_full_graph__s1_api__f_8b_vllm.sh` |
| `run_full_110.sh` | `run_full_graph__s1_api__f_32b_vllm.sh` |
| `run_full_120.sh` | `run_full_graph__s1_api__f_8b_api.sh` |

Each wrapper delegates to `run_step_proof_variant.sh` with a matching YAML file
under `../configs/variants/`. Reuse relationships are now declared in those
variant YAML files instead of hidden in shell arrays.
