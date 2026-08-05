# Runbook: DeepSeek V4 Flash 0731 — DSpark NVFP4 (MTP-5, 1M ctx) on 2× DGX Spark

**Model:** `deepseek-ai/DeepSeek-V4-Flash-0731` @ revision `9e165c30` (~154 GB, 48 shards, deepseek_v4 arch, YaRN 1M ceiling)
**Hardware:** 2× NVIDIA DGX Spark (GB10, 128 GB unified) — head `9105` (10.10.10.1), worker `bdea` (10.10.10.2), RoCE fabric
**Performance:** 84 tok/s decode peak / 67.9 mean · 217 tok/s aggregate @ c6 · 2,680 tok/s prefill @ 100K depth (Aug 5, 2026)
**Quality:** GSM8K 97.0% · MBPP 87.0% · IFEval 86.0% · MMLU-STEM 86.7% · ARC-C 52.0% · HellaSwag 71.0% (Aug 1, 2026, n=100)
**Context:** 1,048,576 tokens (true YaRN ceiling)
**Endpoint:** `http://<head>:8888/v1`, served name `deepseek-v4-flash-dspark`
**Deployment dir:** `~/deepseek-v4-flash-dspark` (clone of `tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`)
**Image:** `vllm-dspark-runtime:dspark-nvfp4-stage-c` (vLLM 0.21.1rc1 fork, DSpark spec decode, NVFP4 `nvfp4_ds_mla` KV cache, B12X MoE kernels)

This is the **production deployment** on the cluster (supersedes the plain FP8 jasl-fork launch, which did ~30–37 tok/s).

---

## What makes it fast

- **DSpark speculative decoding, k=5** — draft heads `mtp.0–2` are the production drafter (overall acceptance ~85–88% of the practical ceiling; position-0 acceptance is content-driven: ~78% structured/repetitive, ~69% code, ~34% prose reasoning).
- **NVFP4 MLA KV cache** — big KV pool at 1M context.
- **Breakable CUDA graphs OFF** — `VLLM_USE_BREAKABLE_CUDAGRAPH=0` gives +28% decode (measured 74.5→95.9 tok/s warm c1 upstream; we keep the opt-out).

## Prerequisites

- 2× DGX Spark on a working RoCE fabric with persistent fabric IPs (see Networking section)
- HF cache with the 0731 checkpoint on **both** nodes (~154 GB each, revision `9e165c30` — exact revision from MiaAI-Lab PR #14)
- Base image `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` pulled on **both** nodes (parallel `docker pull`, ~22.7 GB)
- Docker compose deployment (compose is load-bearing — see pitfall below)

## Step-by-step

### 1. Fabric IPs persistent (one-time per node)

`ip addr add` does NOT survive reboot — use a systemd unit (`/etc/systemd/system/fabric-network.service`) with `|| true` guards; adjust IPs per node (9105=.1, bdea=.2). See the `dgx-spark-fabric` repo for the full fabric setup.

### 2. Download the model (both nodes)

Do **not** pass `cache_dir` to `snapshot_download` — the default honors `HF_HOME` → `~/.cache/huggingface/hub/`. If you did and files landed in `~/.cache/huggingface/models--…`:

```bash
mkdir -p ~/.cache/huggingface/hub
ln -sfn ../models--deepseek-ai--DeepSeek-V4-Flash-0731 \
  ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731
```

Verify shard completeness (48 shards expected, missing must be `[]`):

```python
import json; from pathlib import Path
p = Path("~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/9e165c30").expanduser()
idx = json.loads((p/"model.safetensors.index.json").read_text())
missing = [n for n in set(idx["weight_map"].values()) if not (p/n).exists()]
```

Transfer between Sparks over the fabric (rsync daemon mode, 330+ MB/s; never Tailscale — 28 MB/s).

### 3. Build the runtime image

```bash
cd ~/deepseek-v4-flash-dspark
./build-dspark-vllm-runtime.sh
```

Not as heavy as it looks: base image + 4 stages of **pure Python file patches** (<1 s each once base is cached). Patch 4 (shared-expert `gate_up_proj`, required for 0731) is already baked into `recipe/overlay/vllm/v1/spec_decode/dspark.py` — do NOT bind-mount the patch file. Build auto-rsyncs to `WORKER_HOST` and rebuilds there. Verify: `docker image inspect vllm-dspark-runtime:dspark-nvfp4-stage-c` on both nodes.

### 4. Configure `.env.dspark` (live production config, Aug 5 2026)

```bash
WORKER_HOST=100.82.15.7              # Tailscale IP — SSH/rsync transport (fabric IP refuses SSH)
WORKER_SCRIPT_DIR=/home/vikassridhar/deepseek-v4-flash-dspark
MASTER_ADDR=10.10.10.1
MASTER_PORT=25000
NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0  # dual-rail
NCCL_SOCKET_IFNAME=enp1s0f0np0
NCCL_IB_GID_INDEX=4
NCCL_CROSS_NIC=1
DSPARK_MODEL=deepseek-ai/DeepSeek-V4-Flash-0731
VLLM_PORT=8888
SERVED_MODEL_NAME=deepseek-v4-flash-dspark
VLLM_HOST_IP=10.10.10.1
WORKER_VLLM_HOST_IP=10.10.10.2
DSPARK_VLLM_IMAGE=vllm-dspark-runtime:dspark-nvfp4-stage-c
MAX_MODEL_LEN=1048576                # 1M = YaRN ceiling
MAX_NUM_SEQS=6
GPU_MEMORY_UTILIZATION=0.85
MTP_NUM_TOKENS=***                  # k=5 — locked, see below
```

Also required in `docker-compose.dspark.yml` environment:

```yaml
GLOO_SOCKET_IFNAME: "${NCCL_SOCKET_IFNAME}"
TP_SOCKET_IFNAME: "${NCCL_SOCKET_IFNAME}"
```

### 5. Launch & verify

```bash
cd ~/deepseek-v4-flash-dspark
./start-deepseek-v4-flash-dspark.sh      # worker-first via SSH+compose, waits for :8888/v1/models, smoke-tests
./status-deepseek-v4-flash-dspark.sh
./stop-deepseek-v4-flash-dspark.sh
```

**Cold-start CUDA-graph compile is brutal:** `MAX_NUM_SEQS × (k+1)` capture sizes at 1M context = 35+ minutes of silent compile. Reduce `MAX_NUM_SEQS` or `MAX_MODEL_LEN` for faster dev startup. Watch for `Application startup complete`.

```bash
curl -s http://10.10.10.1:8888/v1/models | python3 -m json.tool
```

### 6. Benchmark (the repo's standard harness)

```bash
cd ~/deepseek-v4-flash-dspark/benchmarks
URL=http://10.10.10.1:8888/v1 MODEL=deepseek-v4-flash-dspark TAG=baseline python3 bench_full.py
```

`bench_full.py` = decode by content type (count300/mult12/json60/bst/story) + concurrency c1–c6 + prefill at depth, temp 0, warm, best-of-2.

**Validated numbers (Aug 5, 2026, baseline restore-verify):**

| Metric | Value |
|---|---|
| Decode peak (count300) | 84.5 tok/s |
| Decode mean | 67.9 tok/s |
| c6 aggregate | 218.7 tok/s |
| Prefill @ 100K depth | 2,678 tok/s |

Acceptance ceiling is content-driven — **never report a single headline tok/s without the content mix**. Story prompts run ~31 tok/s vs count300 84 on identical config; that gap is draft quality, not a bug.

---

## ⚠️ Hard-won pitfalls

1. **k must be 5.** `dspark_block_size=5` in the checkpoint. k=7 boot-rejects, k=10 crashes, k=3 costs ~24% decode.
2. **`VLLM_USE_BREAKABLE_CUDAGRAPH=0`** — breakable graphs cost −28% decode here (MiaAI-Lab PR #14 opt-out).
3. **Benchmark with `stream: false` ONLY.** vLLM emits ≤1 SSE chunk per decode STEP carrying all accepted tokens — counting streamed deltas measures steps/s, not tokens/s (measured 14.7 vs 60.1 tok/s on the identical request). Use `usage.completion_tokens` or server-side `vllm:generation_tokens_total` ÷ wall time.
4. **NEVER hand-recreate with `docker run`.** Compose's `privileged: true`, `gpus: all`, `ulimits.memlock: -1`, `shm_size: 64gb` are load-bearing; hand-rolled runs crash with `Failed to infer device type` / "No CUDA runtime is found". To change env vars, edit compose + `.env.dspark`, then stop/start scripts.
5. **`privileged: true` required for `/dev/infiniband`** — stock compose omits it; without it NCCL/RoCE fails silently at `torch.distributed.new_group`.
6. **Gloo picks the wrong interface** inside the container (`enP7s7`, no IP) → set `GLOO_SOCKET_IFNAME` + `TP_SOCKET_IFNAME` (NCCL_SOCKET_IFNAME only affects NCCL).
7. **NaN-logprobs server bug:** vLLM can't JSON-serialize NaN logprobs on echo+logprobs requests → HTTP 400. Chat/completions traffic unaffected; lm_eval loglikelihood tasks lose some requests.
8. **"Slower than claimed?" → check prompt shape FIRST.** Run `"Count from 1 to 300, separated by commas."` temp=0. ~84 tok/s = healthy; ~35 tok/s even on that = real problem (check GID, privileged, patch 4).
9. **Disk:** 0731 adds ~154 GB next to the FP8 149 GB — prune `docker system df` build cache first on tight nodes.
10. **Bandwidth scheduling:** HF download + 22 GB docker pull on the same WAN starve each other — HF pulls are resumable (`.incomplete` survives kill); sequence them.

## Autoresearch results (Aug 5, 2026) — config is optimal, don't re-test

0/5 env-lever experiments beat baseline (independently reproducing the upstream repo's Jul 29 finding):

| Lever | Result |
|---|---|
| `B12X_W4A16_TC_DECODE=1` | −13% decode — REJECTED |
| `VLLM_DSV4_B12X_COMPRESSED_MLA=1` | −98% — REJECTED |
| `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=1` | flat peak, −8% c6 — REJECTED |
| `VLLM_DSPARK_EXPORT_DRAFT_PROBS=1` | flat, −5% c6 — REJECTED |
| `VLLM_DSPARK_CONFIDENCE_SCHEDULER=hardware` | −10% c6 — REJECTED |

Also settled upstream (do not re-test): draft_sample_method no-op for DSpark; max-model-len 1M→200K / seqs 6→2 / capture-size 36 all neutral; MTU 9000 already set on fabric.

Where remaining decode speed can come from (not flags): better drafter / higher-quality draft heads (acceptance is content-driven), or newer kernel stack / vLLM fork changes (bakeoff showed the kernel stack is the ~9% gap vs vLLM 0.25.2).

## Compared: same 2-node hardware

| Deployment | Decode | Context |
|---|---|---|
| **0731 DSpark NVFP4 (this)** | **84 peak / 68 mean** | **1M** |
| Plain FP8 (jasl fork, MTP-2) | ~30–37 | 200K |
| MiniMax M2.7 AWQ | ~32 | 131K |

## Source

Deployment repo: `tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark` (cloned to `~/deepseek-v4-flash-dspark`). Benchmark parity target: MiaAI-Lab PR #14 (same checkpoint rev, 1M ctx, nvfp4_ds_mla KV, MTP-5).
