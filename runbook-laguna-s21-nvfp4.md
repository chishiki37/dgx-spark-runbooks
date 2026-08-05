# Runbook: Laguna S 2.1 NVFP4 on DGX Spark (1-node and 2-node)

**Model:** poolside/Laguna-S-2.1-NVFP4 (118B MoE / ~8B active, 256 experts top-10 + 1 shared, 1M ctx capped to 262144)
**Current revision:** `07614121` (2026-07-22) — requantized weights + **thinking enabled by default**
**Hardware:** 1× or 2× NVIDIA DGX Spark (GB10)
**Container:** `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` (0.23.1rc1+, native `LagunaForCausalLM` + `DFlashLagunaForCausalLM`)
**License:** OpenMDW-1.1 (commercial OK)
**Validated:** single-Spark 2026-07-22 (rev ea641552); re-validated on rev 07614121 2026-07-24

---

## ⚠️ Revision 07614121 (2026-07-22) — what changed

Poolside pushed a new revision that changes runtime behavior:

1. **Requantized weights** (14→15 shards, 71.9→67 GiB):
   - Observer: `static_minmax` → `minmax`
   - MoE expert layers (`gate_proj`/`up_proj`/`down_proj`) now included in quantization
2. **Thinking is now ON by default** in the chat template (`enable_thinking` default `false`→`true`, plus a new `preserve_thinking` flag). The model now emits long inline "Okay, so..." reasoning in `content` before answering — same failure mode as Qwen3.6.
3. New DFlash draft revision `b0486d15` (pulled automatically by the speculative-config).

**Operational impact:**
- **Benchmarks MUST pass `--gen_kwargs max_gen_toks=1024`** or generative tasks (GSM8K especially) truncate mid-thought and score far below true ability. Already set in the Cluster Manager `bench.sh`.
- Expect lower raw tok/s on generative tasks than the old non-thinking revision (reasoning-token burn is real work, not overhead).
- To get old-style direct answers, pass `chat_template_kwargs: {"enable_thinking": false}` per request.

## Why this model

Best code quality per GB on the cluster. Official poolside numbers: TB-2.1 70.2, SWE-bench Multi 78.5, SWE-Pro 59.4 — Hy3-class agentic coding at 2.5× fewer active params. Our measured head-to-head vs Hy3 295B:

| Metric | Laguna 1×Spark (old rev, no-think) | Laguna TP=2 | Hy3 TP=2 |
|---|---|---|---|
| Throughput (coding, temp=0) | 40.8 tok/s | 52–68 tok/s | 22–23 tok/s |
| HumanEval fenced | 97% | — | 76% |
| MBPP | 85% | — | 72% |
| GSM8K strict | 93% | — | 83% |
| IFEval strict | 78% | — | **90%** |
| MMLU-STEM | 76.1% | — | **82.4%** |

> Note: these are old-revision (non-thinking) numbers. New-revision (07614121) benchmark results are harvested into the Cluster Manager — see the Insights view for live figures.

**Use Laguna for coding/math/speed. Use Hy3 when instruction-following or knowledge MCQs matter more.**

## Model files

- Weights: **67 GiB** (15 shards) as of rev 07614121 — was 71.9 GB / 14 shards on the old revision. ⚠️ HF API `?blobs=true` double-counts xet blobs; verify real size with on-disk `du`.
- DFlash draft model: `poolside/Laguna-S-2.1-DFlash` (2.2 GB BF16) — needed for speculative decoding.

```bash
hf download poolside/Laguna-S-2.1-NVFP4
hf download poolside/Laguna-S-2.1-DFlash
# TP=2: rsync both to the second node at the same path
# rsync -a ~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/ \
#   edgexpert-bdea:~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-NVFP4/
```

## Option A — Single Spark (simplest, 40.8 tok/s)

No CX7, no Ray, no NCCL. Same recipe as TP=2 minus the distributed flags:

```bash
sparkrun run ~/laguna-s-nvfp4.yaml --tp 1 --port 8600 --no-follow
```

Or direct docker + vLLM:

```bash
docker run -d --name laguna --gpus all --network host --ipc host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest \
  vllm serve poolside/Laguna-S-2.1-NVFP4 \
    --served-model-name laguna \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 262144 \
    --kv-cache-dtype fp8_e4m3 \
    --moe-backend marlin \
    --speculative-config '{"model":"poolside/Laguna-S-2.1-DFlash","num_speculative_tokens":7,"method":"dflash"}' \
    --tool-call-parser poolside_v1 --reasoning-parser poolside_v1 --enable-auto-tool-choice \
    --host 0.0.0.0 --port 8600
```

Memory: 72 GB weights at GMU 0.90 leaves ~35 GB for KV — comfortable (SWA-heavy layout, 36/48 layers are sliding-window 512).

Weight load: ~9 min (14 shards, ~40 s/shard).

## Option B — TP=2 across 2 Sparks (52–68 tok/s)

Same as Option A but:

- `--tensor-parallel-size 2 --distributed-executor-backend ray`
- NCCL env per cluster standard: `NCCL_IB_HCA=rocep1s0f`, `NCCL_IB_GID_INDEX=3` (or Socket fallback per MiMo runbook)
- Model at identical path on both nodes
- CX7 IPs set (10.10.10.1 / 10.10.10.2)

Recipe template: `templates/laguna-s-nvfp4.yaml` in spark-vllm-docker.

## Critical flags

| Flag | Why |
|---|---|
| `--moe-backend marlin` | Same GB10 constraint as Hy3 — FlashInfer FP4 path is broken on sm121 |
| DFlash `--speculative-config` | 7 draft tokens via the DFlash head — this is how you get 40+ tok/s single-node |
| `--tool-call-parser poolside_v1` | Native parser; without it OpenWebUI tool calls 400 |
| Thinking toggle | On per-request via `chat_template_kwargs.enable_thinking`. **Disable for benchmarks** — reasoning-token burn makes tok/s numbers non-comparable |

## Benchmark pitfall (HumanEval)

Laguna is a **prose-then-code** model. `humaneval_chat` with stop-at-fence scores **0.03** (stops before any code). MUST use `humaneval_fenced` (no stop tokens + `build_predictions_fenced`) — see `references/lm-eval-vllm-harness.md`.

## Troubleshooting

- **Slow single-stream (<20 tok/s):** DFlash draft model not loaded → check startup log for the speculative config line.
- **400 `"auto" tool choice`:** launched without `--enable-auto-tool-choice --tool-call-parser poolside_v1`.
- **OOM at GMU 0.90:** something else on the GPU (KDE on bdea eats ~12 GB → `sudo systemctl stop sddm`).

## References

- Model: https://huggingface.co/poolside/Laguna-S-2.1-NVFP4 (+ `-DFlash` draft)
- Bring-up notes + quant size analysis: `references/laguna-s-2.1.md` (dgx-spark skill)
