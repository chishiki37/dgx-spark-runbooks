# Runbook: Qwen3.6-35B-A3B on DGX Spark (NVFP4 / FP8)

**Model:** `nvidia/Qwen3.6-35B-A3B-NVFP4` (primary) · `Qwen/Qwen3.6-35B-A3B-FP8` (variant)
**Hardware:** NVIDIA DGX Spark (GB10) — single node (24 GB weights) or TP=2
**Performance:** ~42–58 tok/s decode · smallest and fastest vLLM model on the cluster · instant bring-up (fleet report, Jul 24 2026)
**Context:** 262,144 tokens
**Container:** `vllm-node` (standard `eugr/spark-vllm-docker` image)
**Recipes:** `recipes/qwen3.6-35b-a3b-nvfp4.yaml` · `recipes/qwen3.6-35b-a3b-fp8.yaml` (+ `-no-mtp` variants)

---

## Which variant

| Variant | Weights | Notes |
|---|---|---|
| **NVFP4 (nvidia/)** | ~24 GB | Marlin MoE backend, MTP-3 spec decode — default choice |
| FP8 (Qwen/) | larger | needs `mods/fix-qwen3.6-chat-template` applied |

## Launch (NVFP4, recipe)

```bash
cd ~/spark-vllm-docker
hf download nvidia/Qwen3.6-35B-A3B-NVFP4
./run-recipe.sh qwen3.6-35b-a3b-nvfp4 --name vllm_qwen36
```

Key serve flags (from the recipe):

```
--kv-cache-dtype fp8
--attention-backend flashinfer
--moe-backend marlin                        # env: VLLM_MARLIN_USE_ATOMIC_ADD=1
--enable-chunked-prefill --async-scheduling --enable-prefix-caching
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
--load-format fastsafetensors
--reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
```

Defaults: TP=2, gpu_memory_utilization 0.4 (NVFP4) / 0.8 (FP8), max_model_len 262144, max_num_seqs 4, max_num_batched_tokens 8192. Single-node serving works fine (recipe ships TP=2 + Ray for symmetric cluster use; override with `--tp 1 --no-ray` for solo).

## Verify

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"nvidia/Qwen3.6-35B-A3B-NVFP4","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":32}'
```

## Benchmark behavior — read before judging quality

Fleet benchmark results (Jul 24, n=100, lm_eval):

| Task | Score | Note |
|---|---|---|
| HellaSwag (acc_norm) | **77%** | best on the cluster |
| MBPP | 70% | |
| ARC-C | 53% | |
| GSM8K | 42% ⚠️ | depressed by inline-thinking preamble |
| IFEval | 17% ⚠️ | same artifact |

**The GSM8K/IFEval numbers are measurement artifacts of the thinking preamble, not model weakness on the tasks per se.** Benchmarks need `max_gen_toks ≥ 1024` to avoid truncation artifacts.

**Takeaway (fleet report):** right tool for high-throughput simple tasks — chat, summarization, commonsense QA. Don't reach for it for multi-step math or strict-format work; use MiMo V2.5 / DS4 Flash / Laguna for those.

## Pitfalls

- **MTP off variant** (`-no-mtp`) exists for A/B against spec decode — MoE + triton MTP backend is the fast path here.
- **Chat-template fix mod is FP8-only** — NVFP4 (nvidia) does not need `mods/fix-qwen3.6-chat-template`.
- For the Qwen3.5 predecessor (`Qwen/Qwen3.5-35B-A3B`), recipes `qwen3.5-35b-a3b-fp8.yaml` and the 122B/397B siblings exist in the same dir — same launch pattern.
