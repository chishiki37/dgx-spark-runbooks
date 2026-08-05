# Runbook: Tencent Hy3 295B NVFP4 on 2× DGX Spark

**Model:** kodelow/Hy3-NVFP4-W4A16 (295B MoE / 21B active, 192 experts top-8, 256K ctx)
**Hardware:** 2× NVIDIA DGX Spark (GB10, 128 GB unified memory each), CX7 direct link
**Performance:** ~21.8 tok/s single-stream, ~59.7 tok/s aggregate at 6-way concurrency
**Context:** 128K validated (256K single-sequence is a stretch)
**Container:** `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`
**Recipe:** `~/hy3-nvfp4.yaml` (sparkrun) — validated Jul 2026

---

## Prerequisites

- 2× DGX Spark connected via QSFP56 DAC (CX7 direct link), IPs 10.10.10.1/10.10.10.2
- Model downloaded on ONE node + rsynced to the other (181 GB — don't download twice)
- ~190 GB free per node
- vLLM ≥0.23 (has native `hy_v3` / `hy_v3_mtp` arch support)

## Cluster Details

- **Head:** edgexpert-9105 — CX7 `enp1s0f1np1` → 10.10.10.1
- **Worker:** edgexpert-bdea — CX7 `enp1s0f0np0` → 10.10.10.2
- CX7 IPs don't survive reboots → re-add after every reboot (see MiMo/MiniMax runbooks)

## Step-by-Step

### 1. Download + sync the model (first time only)

```bash
# On bdea (more disk)
hf download kodelow/Hy3-NVFP4-W4A16 --local-dir ~/models/hy3-nvfp4-w4a16

# Rsync to 9105 over CX7 (~460 MB/s, ~7 min)
rsync -avP ~/models/hy3-nvfp4-w4a16/ vikassridhar@10.10.10.1:~/models/hy3-nvfp4-w4a16/
```

Both nodes MUST have the model at the same local path (container bind-mount).

### 2. Fix the tool/reasoning parser token suffixes (CRITICAL)

This checkpoint's special tokens carry a `:opensource` suffix (`<think:opensource>`, `<tool_calls:opensource>`), but vLLM's `hy_v3` parsers hardcode the bare forms. Without this fix:
- Parsers ON → startup crash: `HYV3ReasoningParser could not locate think start/end tokens`
- Parsers OFF → fake plain-text tool calls in `content` instead of structured `tool_calls`

```bash
RP=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")/reasoning/hy_v3_reasoning_parser.py
TP=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")/tool_parsers/hy_v3_tool_parser.py

sed -i 's|"<think>"|"<think:opensource>"|g; s|"</think>"|"</think:opensource>"|g' "$RP"
sed -i 's|"<tool_calls>"|"<tool_calls:opensource>"|g; s|"</tool_calls>"|"</tool_calls:opensource>"|g' "$TP"
```

(Apply inside the container, or bake into the image. Full 9-token patch: see tonyd2wild's repo below.)

### 3. Launch

```bash
HF_HOME=~/hf-cache-writable sparkrun run ~/hy3-nvfp4.yaml --tp 2 --port 8600 --no-follow
```

Recipe essentials (`~/hy3-nvfp4.yaml`):

```yaml
model: /models/hy3-nvfp4-w4a16          # bind-mounted local path
container: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
cluster_only: true
defaults:
  port: 8600
  tensor_parallel: 2
  gpu_memory_utilization: 0.90          # 0.92 FAILS — only ~111.4 GiB free at boot
  max_model_len: 131072
  max_num_seqs: 6
  kv_cache_dtype: fp8_e4m3
  moe_backend: marlin                   # FlashInfer native-FP4 FREEZES GB10s
  enforce_eager: "1"                    # CUDA graphs COST ~25% throughput on sm121
  speculative_config: '{"method":"mtp","num_speculative_tokens":1}'  # spec-2 is NET NEGATIVE
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: "1"
  NCCL_IB_HCA: "rocep1s0f"              # Card 1 only (lowercase 'p'); Card 2 has no valid GID
  NCCL_IB_GID_INDEX: "3"
runtime: vllm-distributed
```

Plus `--tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice` in the command (after the parser fix).

Startup: ~10 min (181 GB weight load across 2 nodes).

### 4. Verify

```bash
curl -s http://10.10.10.1:8600/v1/models
curl -s http://10.10.10.1:8600/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"hy3","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

Tool-call smoke test: a weather function should return `finish_reason: tool_calls` with a clean `tool_calls` object.

### 5. Stop

Bounce BOTH containers between serve attempts — killing vLLM on the head leaves the worker's RayWorkerProc holding ~90 GB; the next serve hangs silently waiting for GPU room.

## Performance Reference (validated Jul 2026)

| Config | Single-stream | 6-way concurrent |
|---|---|---|
| **enforce-eager + MTP spec-1** | **21.8 tok/s** | **59.7 tok/s agg** |
| CUDA graphs + MTP spec-1 | 15.5–16.3 tok/s | — |
| CUDA graphs + MTP spec-2 | 15–16 tok/s | — |

Quality (n=100, temp=0): HumanEval fenced 76% (corrected), MBPP 72%, GSM8K 91/83%, IFEval strict 90%, MMLU-STEM 82.4%, ARC-C 72%.

## Counter-Intuitive Findings (don't "fix" these)

1. **`--enforce-eager` WINS.** CUDA graphs + inductor compile cost ~25% throughput with the marlin W4A16 decode path on sm121.
2. **MTP spec-1, NOT spec-2.** Position-1 draft acceptance 62–76%; position-2 only ~18–21%. Spec-2 pays draft+verify for a token thrown away 4/5 times → net ~30% LOSS.
3. **`--moe-backend marlin` required.** FlashInfer native-FP4 path freezes GB10s on sm121.
4. **GMU 0.90, not 0.92** — boot free memory is only ~111.4/121.69 GiB.
5. **TP=3 impossible** — 8 KV heads don't divide by 3. 2-Spark or 4-Spark only.
6. **Chinese bleed:** Hy3 is Chinese-base; without a system prompt it answers bilingually. Set an English-only system prompt in OpenWebUI.

## Troubleshooting

- **Hang on second launch:** stale worker holding GPU → `docker rm -f` both containers.
- **`World size (2) larger than available GPUs (1)`:** missing `--distributed-executor-backend ray`.
- **Parser crash at startup:** token suffix fix not applied (Step 2).
- **Slow (~15 tok/s):** accidentally left CUDA graphs on → add `--enforce-eager`.

## References

- Quant: https://huggingface.co/kodelow/Hy3-NVFP4-W4A16
- Tony D's recipe + scripts: https://github.com/tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark
- Forum: https://forums.developer.nvidia.com/t/hy3-295b-hunyuan-3-nvfp4-w4a16-mtp-speculative-on-2x-dgx-spark-gb10-128k-ctx-21-8-tok-s-single-59-7-tok-s-6-way/375851
