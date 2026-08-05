# Runbook: MiMo-V2.5 NVFP4 on 2× DGX Spark

**Model:** lukealonso/MiMo-V2.5-NVFP4 (310B MoE, NVFP4 quantized)
**Hardware:** 2× NVIDIA DGX Spark (GB10, 121 GB unified memory each)
**Performance:** ~19 tok/s, 20ms TTFB (streaming)
**Context:** 65,536 tokens (65K)
**Concurrency:** 5× (5 simultaneous requests at 65K each)
**Recipe:** `recipes/mimo-v2.5-nvfp4.yaml`

---

## Prerequisites

- 2× DGX Spark nodes connected via QSFP56 DAC cable (CX7 direct link)
- Both nodes have the sparkrun cluster configured (`dgx-cluster`)
- HF cache with model downloaded on spark2 (sparkrun auto-syncs to gb10)
- Docker image `vllm-node-mimo-v25-nvfp4` built (or will be built from recipe)
- `sudo` access on both nodes (CX7 IPs are lost on reboot)

## Cluster Details

| Role | Hostname | Tailscale IP | CX7 IP | CX7 Interface |
|------|----------|-------------|--------|---------------|
| Head | spark2 | 100.127.212.61 | 10.10.10.1 | `enp1s0f1np1` |
| Worker | gb10 | 100.82.15.7 | 10.10.10.2 | `enp1s0f0np0` |

## Step-by-Step

### 1. Set CX7 IP Addresses (AFTER every reboot)

The CX7 interfaces lose their IPs on reboot. You must re-add them manually.

**On spark2:**
```bash
sudo ip addr add 10.10.10.1/24 dev enp1s0f1np1
```

**On gb10 (SSH from spark2):**
```bash
ssh vikassridhar@100.82.15.7 "sudo ip addr add 10.10.10.2/24 dev enp1s0f0np0"
```

Verify:
```bash
ping -c 3 10.10.10.2   # from spark2 → gb10 over CX7
```

### 2. Download the Model (first time only)

Model is ~171 GB. Downloads to `~/.cache/huggingface/hub/`.

```bash
cd ~/spark-vllm-docker
hf download lukealonso/MiMo-V2.5-NVFP4 --quiet
```

> **If download stalls:** Kill all hf processes (`pkill -9 -f 'hf download'`), remove stale locks (`rm -rf ~/.cache/huggingface/hub/.locks/models--lukealonso--MiMo-V2.5-NVFP4/`), then restart.

### 3. Launch the Server

```bash
cd ~/spark-vllm-docker
uvx sparkrun run recipes/mimo-v2.5-nvfp4.yaml --cluster dgx-cluster
```

This will:
- Build the `vllm-node-mimo-v25-nvfp4` Docker image (first time, ~10 min)
- Apply runtime mods (`fix-mimo-v2-vllm`, `fix-modelopt-mixed-mxfp8`, `drop-caches`)
- Sync model to gb10 (auto, ~170 GB over CX7 link)
- Start vLLM with TP=2 across both nodes

**Startup takes ~4-5 minutes.** Watch for these markers:
```
Resolved architecture: MiMoV2OmniForCausalLM
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
Using TRITON_ATTN_DIFFKV for attention
Using fp8_e4m3 data type to store kv cache
Uvicorn running on http://0.0.0.0:8000
```

### 4. Verify It's Working

```bash
# Check model is loaded
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Test inference
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MiMo-V2.5-NVFP4","messages":[{"role":"user","content":"Hello! What is 2+2?"}],"max_tokens":100}'
```

Expected: Response with `"content": "2 + 2 = 4"` and `"reasoning": "Simple math question."`

### 5. Connect to OpenWebUI

- **Base URL:** `http://192.168.1.44:8000/v1` (or `http://100.127.212.61:8000/v1` via Tailscale)
- **Model ID:** `MiMo-V2.5-NVFP4`
- **API Key:** anything (e.g., `sk-anything`) — vLLM doesn't enforce auth

### 6. Stop the Server

```bash
cd ~/spark-vllm-docker
uvx sparkrun stop recipes/mimo-v2.5-nvfp4.yaml --cluster dgx-cluster
```

---

## Key Configuration Details

The recipe (`recipes/mimo-v2.5-nvfp4.yaml`) has these critical settings:

**NCCL (networking):**
- `NCCL_NET: Socket` — RoCE/RDMA fails on CX7 direct link (`ibv_modify_qp` error 61)
- `NCCL_IB_DISABLE: "1"` — Disable InfiniBand entirely
- `NCCL_SOCKET_IFNAME: enp1s0f1np1,enp1s0f0np0` — **CRITICAL**: Only use first CX7 port (has IPv4). Second port is IPv6 link-local and will break NCCL with `ncclCommInitRank` errors.

**Memory (the 0.85 sweet spot):**
- `gpu_memory_utilization: 0.85` — Budget: 0.85 × 121 = 103 GiB. Model takes 86 GiB, leaving ~17 GiB for KV + graphs.
- `max_model_len: 65536` — 65K context. KV cache ~10 GiB. MiMo's hybrid attention (39/48 SWA layers) makes KV scale sub-linearly.
- `max_num_batched_tokens: 8192`
- Speculative decoding (MTP) is **disabled** to save ~10 GB memory
- **Concurrency:** 5× at 65K context (329,420 KV cache tokens total)

**vLLM flags:**
- `--load-format instanttensor` — Required for NVFP4 checkpoint
- `--attention-backend triton_attn_diffkv` — MiMo uses differential KV (v_head_dim=128, head_dim=192). FA3/FA4 unavailable on sm_121a.
- `--kv-cache-dtype fp8_e4m3` — FP8 KV cache to save memory
- `--tool-call-parser mimo --reasoning-parser mimo` — Native MiMo tool calling + reasoning

## Troubleshooting

**Ray OOM-kills the TP0 worker during weight loading (Jul 23, 2026):**
- Symptom: `ray.exceptions.OutOfMemoryError: ... exceeds the memory usage threshold` in `ray.get(init_worker_refs)` right after InstantTensor loading starts, even with `RAY_memory_usage_threshold=0.99`. The killed worker's actual RSS is tiny (~5 GB).
- Cause: Ray's memory monitor counts GPU-allocated unified memory as host usage on DGX Spark. Loading 171 GB of weights pushes the node to ~96-99% "memory used" and Ray kills the worker mid-load.
- Fix: set `RAY_memory_monitor_refresh_ms=0` on BOTH head and worker containers (disables worker killing entirely). Validated 2026-07-23 — model loaded 100% (145,276 tensors) and served successfully after this change.
- Also `docker rm -f` both containers between retries; stale `ray::IDLE` actors from a prior job cause `ActorHandleNotFoundError: not valid across Ray sessions` noise on shutdown.

**Host hangs during startup:**
- Memory pressure. Reduce `gpu_memory_utilization` to 0.80 or lower `max_num_batched_tokens`.

**`ncclCommInitRank` failure:**
- Check `NCCL_SOCKET_IFNAME` matches your CX7 interface names exactly.
- Verify CX7 IPs are set (`ip addr show enp1s0f1np1` on spark2, `ip addr show enp1s0f0np0` on gb10).
- Check CX7 link is up: `ping 10.10.10.2` from spark2.

**OOM during KV cache allocation:**
- Model takes ~86 GiB. At 0.85×121=103 GiB budget, ~17 GiB left for KV + CUDA graphs.
- Reduce `max_model_len` to 49152 or 32768 if needed.
- Do NOT increase `gpu_memory_utilization` above 0.85 — host hangs at 0.86+.

**Model not found:**
- Verify model is in HF cache: `ls ~/.cache/huggingface/hub/models--lukealonso--MiMo-V2.5-NVFP4/snapshots/*/model*.safetensors | wc -l` (should be 8+ files)

## Performance Reference

| Metric | Value |
|--------|-------|
| Throughput | ~19.2 tok/s |
| TTFB (streaming) | ~20ms |
| Model size | 86 GiB per node |
| KV cache | ~10 GiB (329K tokens) |
| Concurrency | 5× at 65K context |
| Startup time | ~5-6 minutes |
| Architecture | MoE 310B total / 15B active, top-8/256 experts |
| Quantization | NVFP4 (expert weights) |

## Context Length History

| Context | GPU Util | KV Cache | Concurrency | Status |
|---------|----------|----------|-------------|--------|
| 32,768 | 0.82 | ~13 GiB | ~8× | ✅ original |
| 49,152 | 0.85 | ~13 GiB | ~8× | ✅ worked |
| 65,536 | 0.85 | ~10 GiB | 5× | ✅ **current** |
| 131,072 | — | — | — | ❌ too much for 2× Spark |
