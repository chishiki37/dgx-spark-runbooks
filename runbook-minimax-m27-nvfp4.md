# Runbook: MiniMax-M2.7 NVFP4 on 2× DGX Spark

**Model:** nvidia/MiniMax-M2.7-NVFP4 (456B MoE, ~45B active, NVFP4 quantized)
**Hardware:** 2× NVIDIA DGX Spark (GB10, 121 GB unified memory each)
**Performance:** ~24-26 tok/s, ~5s TTFT
**Context:** 196,608 tokens (196K) — model supports up to 1M natively but VRAM-limited
**Recipe:** `~/.hermes/skills/mlops/dgx-spark/templates/minimax-nvfp4-multinode.yaml`

---

## Prerequisites

- 2× DGX Spark nodes connected via QSFP56 DAC cable (CX7 direct link)
- sparkrun cluster `dgx-cluster` configured
- Docker image `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` pulled on both nodes
- HF cache with model downloaded on both nodes (or sparkrun will auto-sync)
- `sudo` access on both nodes (CX7 IPs are lost on reboot)

## Cluster Details

| Role | Hostname | Tailscale IP | CX7 IP | CX7 Interface |
|------|----------|-------------|--------|---------------|
| Head | spark2 | 100.127.212.61 | 10.10.10.1 | `enp1s0f1np1` |
| Worker | gb10 | 100.82.15.7 | 10.10.10.2 | `enp1s0f0np0` |

## Step-by-Step

### 1. Set CX7 IP Addresses (AFTER every reboot)

The CX7 interfaces lose their IPs on reboot. Re-add them manually.

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

Model is ~131 GB (15 safetensor shards). Downloads to `~/.cache/huggingface/hub/`.

**On spark2:**
```bash
hf download nvidia/MiniMax-M2.7-NVFP4 --quiet
```

**On gb10 (or let sparkrun auto-sync):**
```bash
# Option A: Download directly
ssh vikassridhar@100.82.15.7 "hf download nvidia/MiniMax-M2.7-NVFP4 --quiet"

# Option B: Rsync from spark2 over CX7 link (faster if already downloaded)
rsync -avP ~/.cache/huggingface/hub/models--nvidia--MiniMax-M2.7-NVFP4/ \
  vikassridhar@10.10.10.2:~/.cache/huggingface/hub/models--nvidia--MiniMax-M2.7-NVFP4/
```

> **If download stalls:** Kill all hf processes (`pkill -9 -f 'hf download'`), remove stale locks (`rm -rf ~/.cache/huggingface/hub/.locks/models--nvidia--MiniMax-M2.7-NVFP4/`), then restart. The CLI resumes from where it left off.

Verify on both nodes:
```bash
ls ~/.cache/huggingface/hub/models--nvidia--MiniMax-M2.7-NVFP4/snapshots/*/model*.safetensors | wc -l
# Should return 15
```

### 3. Create the Recipe

Save as `~/minimax-m2.7-nvfp4.yaml`:

```yaml
recipe_version: "1"
name: MiniMax-M2.7-NVFP4
description: MiniMax-M2.7 NVFP4 on 2-node DGX Spark cluster (with NCCL RDMA fix)
model: nvidia/MiniMax-M2.7-NVFP4
container: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
cluster_only: true
skip_model_download: true
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  pipeline_parallel: 1
  gpu_memory_utilization: 0.85
  max_model_len: 196608
  load_format: instanttensor
  reasoning_parser: minimax_m2
  tool_call_parser: minimax_m2
env:
  VLLM_MARLIN_USE_ATOMIC_ADD: "1"
  # CRITICAL: Force NCCL to use Card 1 RDMA device (lowercase 'p' prefix)
  # Card 2 (roceP2p1s0f*) has no IP/GID and will cause ibv_modify_qp failures
  NCCL_IB_HCA: "rocep1s0f"
  NCCL_IB_GID_INDEX: "3"
command: |
  vllm serve {model} \
    --trust-remote-code \
    --gpu-memory-utilization {gpu_memory_utilization} \
    -tp {tensor_parallel} \
    -pp {pipeline_parallel} \
    --max-model-len {max_model_len} \
    --load-format {load_format} \
    --enable-auto-tool-choice \
    --tool-call-parser {tool_call_parser} \
    --reasoning-parser {reasoning_parser} \
    --host {host} \
    --port {port}
runtime: vllm-distributed
```

> **Note:** If RDMA/RoCE fails on your CX7 link (as it does on some setups), switch to Socket transport like the MiMo recipe:
> ```yaml
> env:
>   NCCL_NET: Socket
>   NCCL_IB_DISABLE: "1"
>   NCCL_SOCKET_IFNAME: enp1s0f1np1,enp1s0f0np0
> ```
> Performance drops to ~18-20 tok/s with Socket but is more reliable.

### 4. Launch the Server

```bash
cd ~
uvx sparkrun run minimax-m2.7-nvfp4.yaml --cluster dgx-cluster
```

Startup takes ~3-4 minutes. Watch for:
```
Using FlashInferCutlassMxfp8LinearKernel for MXFP8 GEMM
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend
Uvicorn running on http://0.0.0.0:8000
```

### 5. Verify It's Working

```bash
# Check model is loaded
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Test inference
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/MiniMax-M2.7-NVFP4","messages":[{"role":"user","content":"Hello! What is 2+2?"}],"max_tokens":100}'
```

### 6. Test Tool Calling

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"nvidia/MiniMax-M2.7-NVFP4",
    "messages":[{"role":"user","content":"What is the weather in Bangkok?"}],
    "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
    "max_tokens":200
  }'
```

Expected: `finish_reason: "tool_calls"` with `get_weather({"city": "Bangkok"})`.

### 7. Connect to OpenWebUI

- **Base URL:** `http://192.168.1.44:8000/v1` (or `http://100.127.212.61:8000/v1` via Tailscale)
- **Model ID:** `nvidia/MiniMax-M2.7-NVFP4`
- **API Key:** anything (e.g., `sk-dummy`) — vLLM doesn't enforce auth
- **Context window:** 196,608 tokens

**Client-side tips:**
- Set `reasoning_effort: "none"` for faster responses (reasoning mode halves tok/s)
- Use `chat_template_kwargs: {"enable_thinking": true/false}` to toggle thinking mode
- Known issue: model may emit `reasoning` tokens in streaming even with `enable_thinking: false`

### 8. Stop the Server

```bash
uvx sparkrun stop minimax-m2.7-nvfp4.yaml --cluster dgx-cluster
```

---

## Memory Layout

| Component | Per GPU |
|-----------|---------|
| NVFP4 weights | ~65 GB |
| KV cache (196K ctx) | ~23 GB |
| Runtime overhead | ~12 GB |
| **Total** | ~100 GB |
| **GPU budget** (0.85 × 121) | ~103 GB |

## Key Configuration Details

**NCCL (networking):**
- `NCCL_IB_HCA: "rocep1s0f"` — Forces NCCL to use Card 1 RDMA device only. Card 2 (`roceP2p1s0f*`) has no IP/GID and causes `ibv_modify_qp` failures.
- `NCCL_IB_GID_INDEX: "3"` — Correct GID index for CX7 direct link.
- If RoCE still fails, switch to Socket transport (see recipe note above).

**vLLM flags:**
- `--load-format instanttensor` — Required for NVFP4 checkpoint format
- `--tool-call-parser minimax_m2` — Native MiniMax M2 tool calling (OpenAI-compatible)
- `--reasoning-parser minimax_m2` — Native reasoning/thinking support
- `gpu_memory_utilization: 0.85` — Sweet spot. Model fits with 196K context.

## Troubleshooting

**NCCL `ibv_modify_qp` error:**
- The second CX7 Card (`roceP2p1s0f*`) has no valid GID. Force `NCCL_IB_HCA=rocep1s0f` to use only Card 1.
- If that doesn't work, switch to Socket transport entirely (see recipe note).

**`ncclCommInitRank` failure:**
- Verify CX7 IPs are set after reboot (Step 1).
- Check link: `ping 10.10.10.2` from spark2.

**OOM:**
- Reduce `max_model_len` from 196608 to 131072 or lower.
- Reduce `gpu_memory_utilization` to 0.80.

**Model loads but quality is degraded:**
- Ensure model checkpoint is unmodified. Stripped visual/audio weights or patched configs cause subtle NCCL failures.
- Redownload if needed: `hf download nvidia/MiniMax-M2.7-NVFP4 --force-download`

## Performance Reference

| Metric | Value |
|--------|-------|
| Throughput | ~24-26 tok/s |
| TTFT | ~5 seconds |
| Streaming TTFB | instant |
| Model size | ~131 GB total, ~65 GB per GPU |
| Context | 196K tokens (1M native, VRAM-limited) |
| Tool calling | ✅ Native (OpenAI-compatible) |
| Reasoning/thinking | ✅ Native (may leak in streaming) |
| Architecture | MoE 456B total / ~45B active |
| Quantization | NVFP4 |

## Comparison with Other Models on Same Hardware

| Model | Tok/s | Context | Size |
|-------|-------|---------|------|
| MiniMax-M2.7 NVFP4 | ~24-26 | 196K | 456B MoE |
| MiMo-V2.5 NVFP4 | ~19 | 32K | 310B MoE |
| MiniMax-M2.5 AWQ | ~42 | 131K | Smaller/lower quality |
