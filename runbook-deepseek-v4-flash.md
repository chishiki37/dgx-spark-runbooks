# Runbook: DeepSeek V4 Flash (FP8) on 2× DGX Spark

**Model:** deepseek-ai/DeepSeek-V4-Flash (official FP8, ~149 GB, 46 shards)
**Hardware:** 2× NVIDIA DGX Spark (GB10, 128 GB unified memory each)
**Performance:** ~30 tok/s decode, ~600ms TTFT (short prompts), ~5s TTFT (8K context)
**Context:** 200,000 tokens (200K)
**Recipe:** `~/spark-vllm-docker/recipes/deepseek-v4-flash.yaml`
**Docker Image:** `vllm-node-dsv4`
**vLLM Fork:** [jasl/vllm](https://github.com/jasl/vllm) @ `dda4668b` (`codex/ds4-sm120-min-enable`)

---

## Prerequisites

- 2× DGX Spark nodes connected via QSFP56 200G DAC cable (CX7 direct link)
- Both nodes have the sparkrun cluster configured
- HF cache with model downloaded on spark2 (~149 GB)
- Docker image `vllm-node-dsv4` built on both nodes
- `sudo` access on both nodes (CX7 IPs lost on reboot)
- **SDDM stopped on gb10** — KDE Plasma desktop eats ~12 GB GPU memory

## Cluster Details

| Role | Hostname | Tailscale IP | CX7 IP | CX7 Interface | RoCE Device |
|------|----------|-------------|--------|---------------|-------------|
| Head | spark2 | 100.127.212.61 | 10.10.10.1 | `enp1s0f1np1` | `rocep1s0f1` |
| Worker | gb10 | 100.82.15.7 | 10.10.10.2 | `enp1s0f0np0` | `rocep1s0f0` |

**NIC asymmetry note:** The two nodes use different CX7 ports — spark2 uses port 1, gb10 uses port 0. This is normal for these machines.

## Step-by-Step

### 1. Set CX7 IP Addresses (AFTER every reboot)

The CX7 interfaces lose their IPs on reboot. Re-add them manually.

**On spark2:**
```bash
sudo ip addr add 10.10.10.1/24 dev enp1s0f1np1
```

**On gb10 (SSH from spark2):**
```bash
ssh 100.82.15.7 "sudo ip addr add 10.10.10.2/24 dev enp1s0f0np0"
```

Verify:
```bash
ping -c 3 10.10.10.2   # from spark2 → gb10 over CX7
```

### 2. Stop SDDM on gb10 (every boot)

KDE Plasma uses ~12 GB GPU memory. Must be stopped:
```bash
ssh 100.82.15.7 "sudo systemctl stop sddm"
```

Verify GPU memory freed:
```bash
ssh 100.82.15.7 "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
# Should be <1 GB (just framebuffer)
```

### 3. Download the Model (first time only)

```bash
cd ~/spark-vllm-docker
hf download deepseek-ai/DeepSeek-V4-Flash --quiet
```

Verify (should show 46 safetensor shards):
```bash
ls ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/*/model*.safetensors | wc -l
```

> Model auto-syncs to gb10 via sparkrun when launching.

### 4. Build the Docker Image (first time, or after Dockerfile changes)

```bash
cd ~/spark-vllm-docker
./build-and-copy.sh \
  --vllm-repo https://github.com/jasl/vllm.git \
  --vllm-ref dda4668b59567416f86956cfe7bbc1eab371a61e \
  --rebuild-vllm \
  -t vllm-node-dsv4 \
  -c
```

Takes ~6-7 minutes: NCCL compile (3 min) + vLLM build (1 min) + copy to gb10 over CX7 (2.5 min).

> **Why the pinned commit?** The `jasl/vllm` fork has GB10-specific validation at this commit. Branch aliases may drift.

### 5. Launch the Server

```bash
cd ~/spark-vllm-docker
./run-recipe.sh deepseek-v4-flash --no-ray --name vllm_ds4
```

Startup takes ~3-4 minutes. Watch for these markers:
```
Loading safetensors checkpoint shards: 100% (46/46)
MTP draft model loaded: 39 params
Capturing CUDA graphs (decode, FULL): 100%
Uvicorn running on http://0.0.0.0:8000
```

### 6. Verify It's Working

```bash
# Check model loaded
curl -s http://192.168.1.44:8000/v1/models | python3 -m json.tool

# Test inference
curl -s http://192.168.1.44:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Write a short poem about stars."}],"max_tokens":128}'
```

### 7. Connect to OpenWebUI

- **Base URL:** `http://192.168.1.44:8000/v1` (local) or `http://100.127.212.61:8000/v1` (Tailscale)
- **Model ID:** `deepseek-v4-flash`
- **API Key:** anything (vLLM doesn't enforce auth)
- **Context window:** 200,000 tokens

### 8. Stop the Server

```bash
docker stop vllm_ds4
ssh 100.82.15.7 "docker stop vllm_ds4"
```

---

## Key Configuration Details

**MTP (Multi-Token Prediction):**
- `--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":2}'`
- Generates up to 3 tokens per forward pass (1 target + 2 draft)
- Acceptance rate: ~55-68% — gives ~2.3× effective throughput vs no MTP
- **Without MTP, throughput drops to ~14 tok/s**

**Expert Parallel:**
- `--enable-expert-parallel` — optimizes expert routing for MoE architecture
- Workers show as `Worker_TP0_EP0` / `Worker_TP1_EP1` (TP + EP combined)

**NCCL/RDMA:**
- `NCCL_IB_DISABLE: 0` — RoCEv2 over CX7 InfiniBand
- `NCCL_IB_HCA: rocep1s0f1` (spark2) / `rocep1s0f0` (gb10) — set per-node by launch-cluster.sh
- Bandwidth: ~109 Gb/s (line rate), latency: ~1.4 µs
- **Communication is NOT the bottleneck** — GPU memory bandwidth is

**GPU Memory:**
- `gpu_memory_utilization: 0.80` — Model + KV cache fits comfortably
- 0.85 also works but offers no TPS benefit
- Must stop SDDM on gb10 first (saves ~12 GB)

**Other critical flags:**
- `--no-enable-flashinfer-autotune` — prevents runtime JIT latency spikes
- `--kv-cache-dtype fp8` — FP8 KV cache (saves memory vs BF16)
- `--compilation-config {cudagraph: FULL_AND_PIECEWISE, custom_ops: [all]}`
- `--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4`
- `--reasoning-config {reasoning_parser: deepseek_v4, ...}` — native reasoning
- `--default-chat-template-kwargs '{"thinking":true}'` — enables `<think>` blocks

## Memory Layout

| Component | Per GPU |
|-----------|---------|
| FP8 weights | ~75 GB |
| KV cache (200K ctx, FP8) | ~12 GB |
| CUDA graphs + runtime | ~10 GB |
| **Total** | ~97 GB |
| **GPU budget** (0.80 × 128) | ~102 GB |

## Performance Reference

| Metric | Value |
|--------|-------|
| Decode throughput (tg128) | ~30 tok/s |
| Decode throughput (tg256) | ~31 tok/s |
| Decode throughput (tg512) | ~30 tok/s |
| TTFT (short prompt) | ~600 ms |
| Prefill @ 4K | ~387 tok/s |
| Prefill @ 8K | ~889 tok/s |
| MTP acceptance rate | ~55-68% |
| Startup time | ~4 minutes |
| Context | 200,000 tokens |
| Architecture | MoE, FP8 E4M3 128×128 block |
| Model size | ~149 GB (46 shards) |

### TPS History (this setup)

| Change | TPS | Note |
|--------|-----|------|
| Base (no MTP, no EP) | ~24 tok/s | Original recipe |
| +MTP + EP + fixes | ~29-32 tok/s | All flags from NVIDIA forum |
| +NCCL v2.30u1 rebuild | ~30 tok/s | No change — confirms GPU-bound |
| Developer baseline (jasl) | ~35 tok/s | "conversational c=1" |
| Forum claim (tonyd615) | ~44 tok/s | Possibly peak/streaming measurement |

## Troubleshooting

**Server won't start — `ncclCommInitRank` failure:**
- Check CX7 IPs are set (Step 1)
- Verify RDMA link: `ping -c 3 10.10.10.2`
- Run RDMA bandwidth test: `ib_write_bw -d rocep1s0f1 10.10.10.2` (should see ~109 Gb/s)

**OOM / CUDA out of memory:**
- Stop SDDM on gb10 (Step 2) — frees ~12 GB
- Reduce `gpu_memory_utilization` to 0.75
- Check for zombie containers: `docker ps -a`

**Low TPS (<20 tok/s):**
- Verify MTP is active: check for `MTP draft model loaded` in startup logs
- Check `curl http://192.168.1.44:8000/metrics | grep spec_decode` — should show non-zero draft/accepted tokens
- Verify `--no-enable-flashinfer-autotune` is set (prevents JIT spikes)
- Make sure SDDM is off on gb10

**Model loads but empty output / errors:**
- Ensure `--tokenizer-mode deepseek_v4` is set (fallback to gpt2 tokenizer produces garbage)
- Check Docker image was built with pinned vLLM commit, not a branch alias

**CX7 link wedged / ACCESS_REG timeout:**
- Full cold reboot clears it. Nothing else reliably fixes mlx5 wedged state.

## Compared: Other Models on Same Hardware

| Model | Tok/s | Context | Quantization |
|-------|-------|---------|-------------|
| **DeepSeek V4 Flash** | **~30** | **200K** | FP8 |
| MiniMax-M2.7 | ~24-26 | 196K | NVFP4 |
| MiniMax-M2.5 AWQ | ~42 | 131K | AWQ |
| MiMo-V2.5 | ~19 | 65K | NVFP4 |

---

## Reference: NVIDIA Forum Recipe

This runbook is based on the [official NVIDIA forum recipe](https://forums.developer.nvidia.com/t/deepseek-v4-flash-official-fp8-running-across-2x-dgx-spark-tp-2-mtp-200k-ctx-recipe-numbers/370309) by tonyd615, using the [jasl/vllm GB10 fork](https://github.com/jasl/vllm) and [eugr/spark-vllm-docker PR #219](https://github.com/eugr/spark-vllm-docker/pull/219).

Recipe file: `~/spark-vllm-docker/recipes/deepseek-v4-flash.yaml`
