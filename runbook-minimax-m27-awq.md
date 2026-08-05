# Runbook: MiniMax-M2.7 AWQ on 2× DGX Spark

**Model:** cyankiwi/MiniMax-M2.7-AWQ-4bit (456B MoE, AWQ 4-bit quantized)
**Hardware:** 2× NVIDIA DGX Spark (GB10, 128 GB unified memory each)
**Performance:** ~40-42 tok/s decode, fast TTFT
**Context:** 196,608 tokens (196K)
**Recipe:** `~/spark-vllm-docker/recipes/minimax-m2.7-awq.yaml`
**Docker Image:** `vllm-node`

---

## Prerequisites

- 2× DGX Spark nodes connected via QSFP56 200G DAC cable (CX7 direct link)
- Both nodes have the sparkrun cluster configured
- HF cache with model downloaded on spark2 (~122 GB, 27 safetensor shards)
- Docker image `vllm-node` built on both nodes
- `sudo` access on both nodes (CX7 IPs lost on reboot)

## Cluster Details

| Role | Hostname | Tailscale IP | CX7 IP | CX7 Interface | RoCE Device |
|------|----------|-------------|--------|---------------|-------------|
| Head | spark2 | 100.127.212.61 | 10.10.10.1 | `enp1s0f1np1` | `rocep1s0f1` |
| Worker | gb10 | 100.82.15.7 | 10.10.10.2 | `enp1s0f0np0` | `rocep1s0f0` |

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

### 2. Download the Model (first time only)

Model is ~122 GB (27 AWQ safetensor shards). Downloads to `~/.cache/huggingface/hub/`.

```bash
hf download cyankiwi/MiniMax-M2.7-AWQ-4bit --quiet
```

Verify:
```bash
ls ~/.cache/huggingface/hub/models--cyankiwi--MiniMax-M2.7-AWQ-4bit/snapshots/*/model*.safetensors | wc -l
# Should return 27
```

> Model auto-syncs to gb10 via sparkrun when launching.

### 3. Launch the Server

```bash
cd ~/spark-vllm-docker
./run-recipe.sh minimax-m2.7-awq --no-ray --name vllm_mm27
```

Startup takes ~3-4 minutes. Watch for:
```
Loading safetensors checkpoint shards: 100% (27/27)
AWQ linear method loaded
Uvicorn running on http://0.0.0.0:8000
```

> **Alternative: RoCE/RDMA deployment** — For higher throughput (~42 tok/s vs ~20 tok/s on TCP), use the `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` container with `NCCL_IB_HCA="rocep1s0f"` and `NCCL_IB_GID_INDEX="4"`. The Spark Arena benchmark of 42 tok/s uses RoCE.

### 4. Verify It's Working

```bash
# Check model loaded
curl -s http://192.168.1.44:8000/v1/models | python3 -m json.tool

# Test inference
curl -s http://192.168.1.44:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minimax-m2.7","messages":[{"role":"user","content":"Hello! What is 2+2?"}],"max_tokens":100}'
```

### 5. Connect to OpenWebUI

- **Base URL:** `http://192.168.1.44:8000/v1` (local) or `http://100.127.212.61:8000/v1` (Tailscale)
- **Model ID:** `minimax-m2.7`
- **API Key:** anything (vLLM doesn't enforce auth)
- **Context window:** 196,608 tokens

### 6. Stop the Server

```bash
docker stop vllm_mm27
ssh 100.82.15.7 "docker stop vllm_mm27"
```

---

## Key Configuration Details

**AWQ Quantization:**
- 4-bit weight quantization — model is ~122 GB (vs ~131 GB NVFP4)
- Uses `--load-format fastsafetensors` for optimized loading
- No special kernel requirements (unlike NVFP4 which needs `VLLM_MARLIN_USE_ATOMIC_ADD`)

**Distributed Backend:**
- `--distributed-executor-backend ray` — Uses Ray for multi-node orchestration
- TP=2 across 2 nodes — model fits in ~60 GB per GPU

**NCCL/RDMA (for RoCE deployment):**
- `NCCL_IB_HCA: "rocep1s0f"` — **Regex** matching Card 1 only (excludes Card 2 `roceP2p1s0f*`)
- `NCCL_IB_GID_INDEX: "4"` — Correct GID for RoCEv2 `::ffff:10.10.10.x`
- **Note:** The `vllm-node` container image uses TCP/Socket by default. For RoCE speeds, switch to `dgx-vllm-eugr-nightly` container.

**GPU Memory:**
- `gpu_memory_utilization: 0.80` — Minimum for 196K context
- 0.75 causes OOM: "23.25 GiB KV needed, 22.91 GiB available"
- At 0.80: model ~60 GB + KV cache ~23 GB + runtime ~13 GB = ~96 GB

**Tool Calling & Reasoning:**
- `--tool-call-parser minimax_m2` — Native MiniMax M2 tool calling (OpenAI-compatible)
- `--reasoning-parser minimax_m2` — Native reasoning/thinking support
- `--enable-auto-tool-choice` — Model decides when to call tools

## Memory Layout

| Component | Per GPU |
|-----------|---------|
| AWQ weights | ~60 GB |
| KV cache (196K ctx) | ~23 GB |
| Runtime + Ray overhead | ~13 GB |
| **Total** | ~96 GB |
| **GPU budget** (0.80 × 128) | ~102 GB |

## Performance Reference

| Metric | Value (TCP) | Value (RoCE) |
|--------|------------|-------------|
| Decode throughput | ~20 tok/s | ~40-42 tok/s |
| Prefill throughput | ~1,000 tok/s | ~2,800 tok/s |
| TTFT (short prompt) | ~2s | ~1s |
| Context | 196,608 tokens | 196,608 tokens |
| Startup time | ~4 minutes | ~4 minutes |
| Architecture | MoE 456B total / ~45B active | MoE 456B total / ~45B active |
| Quantization | AWQ 4-bit | AWQ 4-bit |

> **TCP vs RoCE:** The `vllm-node` container lacks RoCE/IB drivers, so NCCL falls back to TCP sockets. This halves throughput (~20 tok/s vs ~40 tok/s). The Spark Arena benchmark uses the `eugr-nightly` container with proper RoCE support.

## Comparison: MiniMax AWQ vs NVFP4

| Model | Quant | Tok/s (RoCE) | Context | Size |
|-------|-------|-------------|---------|------|
| MiniMax-M2.7 AWQ | AWQ 4-bit | **~40-42** | 196K | 456B MoE |
| MiniMax-M2.7 NVFP4 | NVFP4 | ~24-26 | 196K | 456B MoE |
| MiniMax-M2.5 AWQ | AWQ 4-bit | ~42 | 131K | 310B MoE |

**Bottom line:** AWQ gives ~65% more throughput than NVFP4 on the same hardware.

## Troubleshooting

**"23.25 GiB KV needed, 22.91 GiB available" OOM:**
- Bump `gpu_memory_utilization` from 0.75 to 0.80
- Or reduce `max_model_len` to 193696

**NCCL `ibv_modify_qp` error (RoCE deployment):**
- Card 2 (`roceP2p1s0f*`) has no valid GID. Use `NCCL_IB_HCA="rocep1s0f"` regex
- Verify GID index: `cat /sys/class/infiniband/rocep1s0f1/ports/1/gids/4` should show `::ffff:10.10.10.1`
- If RoCE still fails, fall back to TCP: `NCCL_NET=Socket NCCL_IB_DISABLE=1`

**FileNotFoundError for safetensors (container):**
- Symlinks to host paths (`~/models/`) break in containers
- Fix: Use hardlinks for large safetensors + direct copies for small config files
- Or set `skip_model_download: true` and let sparkrun auto-sync model files

**Slow throughput (~20 tok/s instead of ~40):**
- Check if using TCP container (`vllm-node`) instead of RoCE container (`eugr-nightly`)
- Verify RDMA link: `ib_write_bw -d rocep1s0f1 10.10.10.2` (should see ~109 Gb/s)
- Check `NCCL_IB_HCA` matches active RoCE device

**Model not found / empty output:**
- Verify model downloaded: `ls ~/.cache/huggingface/hub/models--cyankiwi--MiniMax-M2.7-AWQ-4bit/snapshots/*/model*.safetensors | wc -l` (27 files)
- Check snapshot path: `~/.cache/huggingface/hub/models--cyankiwi--MiniMax-M2.7-AWQ-4bit/snapshots/67a31123781726f0bb186704f7876405ad6496a5`

## RoCE Deployment (Alternative Recipe)

For maximum throughput with RoCE/RDMA, use this recipe instead of the default:

```yaml
recipe_version: "1"
name: MiniMax-M2.7-AWQ-RoCE
model: cyankiwi/MiniMax-M2.7-AWQ-4bit
container: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
cluster_only: true
skip_model_download: true
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  gpu_memory_utilization: 0.75
  max_model_len: 196608
  load_format: fastsafetensors
  served_model_name: minimax-m2.7
env:
  NCCL_IB_HCA: "rocep1s0f"
  NCCL_IB_GID_INDEX: "4"
command: |
  vllm serve {model} \
    --trust-remote-code \
    --gpu-memory-utilization {gpu_memory_utilization} \
    -tp {tensor_parallel} \
    --max-model-len {max_model_len} \
    --load-format {load_format} \
    --enable-auto-tool-choice \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2 \
    --host {host} \
    --port {port} \
    --served-model-name {served_model_name}
```

> **Key difference from default recipe:** Uses `eugr-nightly` container (RoCE drivers), `NCCL_IB_HCA="rocep1s0f"` (Card 1 regex), `NCCL_IB_GID_INDEX="4"` (RoCEv2 GID), and no Ray dependency.
