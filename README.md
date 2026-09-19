# DGX Spark Runbooks

Deployment runbooks for every model we've run on the DGX Spark cluster (6× GB10, 128 GB unified each, MikroTik CRS812 RoCE fabric — see [`dgx-spark-fabric`](https://github.com/chishiki37/dgx-spark-fabric) for the fabric setup). Each runbook is written from an actual validated deployment, not from upstream docs.

**Master strategy:** every new deployment or optimization campaign follows the [End-to-End Model Serving Recipe Playbook](model-serving-end-to-end-playbook.md) — a gated pipeline (scope → provenance audit → baseline capture → fabric proof → correctness → honest context/concurrency validation → optimization ladder → load/soak → package → canary → maintenance) with evidence labels on every external claim. The runbooks below are its outputs.

Companion repos (model-specific, published separately):
- [`glm-5.2-quanttrio-4x-dgx-spark`](https://github.com/chishiki37/glm-5.2-quanttrio-4x-dgx-spark) — GLM-5.2 QuantTrio Int4-Int8Mix on 4× Spark
- [`minimax-h3-av-comfyui-recipe`](https://github.com/chishiki37/minimax-h3-av-comfyui-recipe) — MiniMax-H3 audiovisual generation via ComfyUI
- [`dgx-spark-fabric`](https://github.com/chishiki37/dgx-spark-fabric) — CRS812 RDMA fabric, NCCL bench, vLLM recipes

## Runbook index

| Runbook | Model | Quant | Nodes | Decode | Context |
|---|---|---|---|---|---|
| [DeepSeek V4 Flash 0731 DSpark](runbook-deepseek-v4-flash-0731-dspark.md) ⭐ production | deepseek-ai/DeepSeek-V4-Flash-0731 | NVFP4 + NVFP4 MLA KV | 2 | 84 peak / 68 mean, 217 @ c6 | 1M |
| [DeepSeek V4 Flash (FP8, jasl fork)](runbook-deepseek-v4-flash.md) | deepseek-ai/DeepSeek-V4-Flash | FP8 | 2 | ~30 | 200K |
| [DeepSeek V4 Flash Abliterated](runbook-deepseek-v4-flash-abliterated-dspark.md) | drowzeys/…-DSpark-Abliterated-32-32 (gated) | NVFP4 | 2 | (0731 path) | 1M |
| [MiMo V2.5](runbook-mimo-v25-nvfp4.md) | MiMo-V2.5 309B-A15B | NVFP4 | 2 | ~19 | 32K |
| [Hy3 295B](runbook-hy3-295b-nvfp4.md) | Hunyuan-3 295B-A21B | NVFP4-W4A16 | 2 | ~22 | 128K |
| [Laguna S 2.1](runbook-laguna-s21-nvfp4.md) | poolside/Laguna-S-2.1 118B-A8B | NVFP4 | 1–2 | ~41 (solo) | 262K |
| [MiniMax M2.7 AWQ](runbook-minimax-m27-awq.md) | MiniMax-M2.7 456B-A45B | AWQ | 2 | ~32 | 131K |
| [MiniMax M2.7 NVFP4](runbook-minimax-m27-nvfp4.md) (superseded by AWQ) | MiniMax-M2.7 | NVFP4 | 2 | ~24–26 | 196K |
| [Qwen3.6 35B-A3B](runbook-qwen3.6-35b-a3b.md) | nvidia/Qwen3.6-35B-A3B-NVFP4 | NVFP4 / FP8 | 1 | ~42–58 | 262K |
| [Gemma 4 26B-A4B](runbook-gemma4-26b-a4b.md) | nvidia/Gemma-4-26B-A4B-NVFP4 | NVFP4 | 1 (solo) | not yet benched | 262K |
| [GLM-5.3 EXL3 Abliterated](runbook-glm53-exl3-abliterated.md) | drowzeys/keys-GLM-5.3-EXL3-Abliterated | EXL3 3-bit | 4 (TP4+DCP4) | 12.6 C1 / 39.7–46.5 agg @ C8 | 1M |
| [GLM-5.3 EXL3 Ablit — cuda-exl3 + DFlash2](runbook-glm53-exl3-cuda-exl3-dflash2.md) | drowzeys/keys-GLM-5.3-EXL3-Abliterated | EXL3 3-bit + cuda-exl3 MoE kernels | 4 (TP4, DCP ladder) | **16.4 C1 / 56.4 agg @ C8 (200K/DCP1)**; 13.8 C1 / 41.9 C8 (1M/DCP4) | 200K–1M |
| [GLM-5.3 Int4-Int8Mix NFS](runbook-glm53-int4-int8mix-nfs.md) | vikasclawd/GLM-5.3-Int4-Int8Mix (743B) | Int4-Int8Mix | 4 (TP4+MTP, NFS-o-RDMA weights) | 26.3 C1 / 56.1 C4 / 89.2 C8 agg | 200K |
| [GLM-5.3 Flash NVFP4 + DFlash2](runbook-glm53-flash-dflash2-tp2.md) | RedHatAI/GLM-5.3-Flash-NVFP4 + incoai/GLM-5.3-Flash-DFlash2 | NVFP4 (compressed-tensors W4A4) | 2 (TP2, DFlash2 k5 + CUDA graphs) | **22.7 C1 prose / 32.1 C1 code / 51.2 C4 / 72.4 C8 agg** | 262K |
| [GLM-5.3 Flash NVFP4 MTP](runbook-glm53-flash-nvfp4.md) (superseded by DFlash2 lane; ModelOpt weights corrupt tokens) | LibertAIDAI/GLM-5.3-Flash-NVFP4 | NVFP4 weight-only | 2 or 4 (TP2/TP4 + MTP-3) | see runbook | 262K–1M |
| [DiffusionGemma 26B](runbook-diffusion-gemma-26b.md) | google/diffusiongemma-26B-A4B-it | BF16 | 1 | ~119 avg (diffusion) | — |

`recipes/` contains the matching sparkrun/vLLM recipe YAMLs (from `eugr/spark-vllm-docker`-style deployments).

## Fleet quality snapshot (n=100, lm_eval — see each runbook for detail)

- **Quality all-rounder:** MiMo V2.5 (GSM8K 94, MBPP 86, MMLU-STEM 82.6)
- **Instruction following:** Hy3 (IFEval 90) and DS4 Flash (IFEval 91; 0731 DSpark: GSM8K 97, MMLU-STEM 86.7)
- **Code:** Laguna S 2.1 (HumanEval 97, MBPP 85)
- **Long context / throughput:** DS4 Flash 0731 DSpark (1M ctx, 84 tok/s peak)
- **Fast + tiny:** Qwen3.6 35B-A3B (chat/summarization only — weak on math/strict formats)

## Recipe-only models (run or cached, runbook not yet written)

`qwen3.5-122b-fp8`, `qwen3.5-122b-int4-autoround`, `qwen3.5-397b-int4-autoround`, `qwen3.5-35b-a3b-fp8`, `qwen3.6-35b-a3b-nvfp4-no-mtp`, `qwen3.6-35b-a3b-fp8-dflash`, `qwen3-coder-next-fp8/int4`, `glm-4.7-flash-awq`, `minimax-m2/m2.5-awq`, `nemotron-3-nano/super-nvfp4`, `openai-gpt-oss-120b`, `step-3.7-flash-fp8/nvfp4`, `mimo-v2.5-nvfp4-textonly`. Recipes live in the `spark-vllm-docker` fork on the cluster.

## Cluster notes

- Nodes: 9105 (.1), bdea (.2), 1d49 (.11), 3b24 (.12), ae1e (.13), cb98 (.14) on the 10.10.10.0/24 + 10.10.20.0/24 rails; fabric IPs are systemd-persisted (they do not survive reboot otherwise).
- 1d49 hosts ComfyUI (MiniMax H3) permanently — never schedule serving work there.
- Provenance: fleet reports 2026-07-24 and 2026-08-01; DS4-DSpark autoresearch 2026-08-05.
