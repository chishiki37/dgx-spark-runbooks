# Runbook: Gemma 4 26B-A4B (NVFP4) on DGX Spark

**Model:** `nvidia/Gemma-4-26B-A4B-NVFP4` (block-diffusion-free MoE, 26B total / 4B active)
**Hardware:** NVIDIA DGX Spark (GB10) — **solo node only** (recipe is `solo_only: true`)
**Context:** 262,144 tokens
**Container:** `vllm-node` (standard `eugr/spark-vllm-docker` image; vLLM uses Transformers v5 by default — no legacy TF5 build args needed)
**Recipe:** `recipes/gemma4-26b-a4b-nvfp4.yaml`
**Benchmark status:** not yet measured on the fleet — treat speed numbers as TBD

---

## Launch

```bash
cd ~/spark-vllm-docker
hf download nvidia/Gemma-4-26B-A4B-NVFP4
./run-recipe.sh gemma4-26b-a4b-nvfp4 --name vllm_gemma4
```

Key serve flags (from the recipe):

```
--load-format instanttensor
--enable-prefix-caching
--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
--kv-cache-dtype fp8
--speculative-config '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":4,"moe_backend":"triton"}'
-tp 2 --distributed-executor-backend ray
```

Defaults: gpu_memory_utilization 0.7, max_model_len 262144, max_num_batched_tokens 8192.

## What's notable

- **MTP uses a separate draft model**: `google/gemma-4-26B-A4B-it-assistant` (downloaded alongside; k=4 speculative tokens, triton MoE backend). The assistant model must be in the HF cache or startup fails.
- **Gemma4-native parsers**: `--tool-call-parser gemma4` + `--reasoning-parser gemma4` — a `mods/fix-gemma4-tool-parser` mod exists in the repo (currently commented out in the recipe; re-enable only if tool-call parsing regresses).
- **Instanttensor load format** for fast startup from cache.
- Related smaller family members cached on the cluster: `google/gemma-4-12B-it` (also used as the VoiceClaw LLM), `google/gemma-4-E4B`/`E2B` (on-device class).

## Variants tried on this hardware

HF cache evidence on 9105: `nvidia/Gemma-4-26B-A4B-NVFP4` (this recipe), `google/gemma-4-26B-A4B-it` (bf16 recipe `gemma4-26b-a4b.yaml`), `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4`, `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`. NVFP4 (nvidia) is the maintained recipe.

## Pitfalls

- **Solo only** — the recipe refuses cluster launch; one Spark serves it comfortably (~26B-A4B ≈ small active footprint).
- A dedicated `gemma4-env` venv exists in `~/spark-vllm-docker/gemma4-env/` for transformers-v5 side experiments — not needed for the vLLM recipe path.
- For the diffusion-based sibling see `runbook-diffusion-gemma-26b.md` (different architecture: block-diffusion decoding, custom loop — not this model).
