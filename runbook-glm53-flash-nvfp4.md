# Runbook: GLM-5.3-Flash NVFP4 on 2× and 4× DGX Spark (TP2 / TP4)

- **Model:** zai-org/GLM-5.3-Flash (320B total / 18B active MoE, `glm5_next`, hybrid KDA linear-attn + DeepSeek sparse-attn, native MTP head, 262K ctx) — quant: LibertAIDAI/GLM-5.3-Flash-NVFP4 (weight-only NVFP4, routed-expert FFNs only, 194.7 GB, 120 shards)
- **Hardware:** NVIDIA DGX Spark (GB10, SM121) — TP2 pair: aitopatom-cb98 (head) + edgexpert-3b24 (worker); TP4 adds edgexpert-9105 + edgexpert-bdea. 200G CRS812 RoCEv2 fabric, NFSoRDMA weights from cb98.
- **Engine:** vLLM day-0 image `vllm/vllm-openai:glm53-flash-arm64-cu130` + 7-patch sm121 chain (v1→v8) from tonyd2wild's recipes; final image tag `radixark/vllm-glm53-flash:sm121-v8` (local build).
- **Performance (measured, on-box, prose, temp 0, thinking off):** see Performance Reference below.
- **Endpoint:** `http://<head>:8000/v1` (served name `glm-5.3-flash`)
- **Campaign date:** 2026-08-27

## Prerequisites

1. Four (or two) DGX Sparks with fabric links UP on rail 1 (`ip -br addr` shows 10.10.10.x UP; `ibdev2netdev` shows `rocep1s0f0 ==> enp1s0f0np0 (Up)`).
2. Weights staged on the head node (local NVMe) and NFS-exported read-only to both fabric subnets; workers mount over NFSoRDMA (vers=3, proto=rdma, port 20049) at a path visible to the launcher.
3. Patched image on every rank (`docker save | ssh <node> docker load` over fabric, ~90 s/31 GB). Verify `grep '^IMAGE'` matches on all nodes before every launch.
4. sudo available on all nodes for `drop_caches` + cache-flusher sidecar (GB10 NVRM allocator needs MemFree, not MemAvailable — see Troubleshooting).

## Step-by-step

1. Download `LibertAIDAI/GLM-5.3-Flash-NVFP4` (194.7 GB) to the head node (hfvenv + snapshot_download, ~66 min).
2. NFS-export the models dir to `10.10.10.0/24` + `10.10.20.0/24` (ro,no_root_squash); enable `rdma 20049` in `/proc/fs/nfsd/portlist` AFTER the final nfs-server restart (restart clears it).
3. On each worker: `sunrpc.tcp_slot_table_entries=256` sysctl, mount `vers=3,proto=rdma,port=20049`, verify `nfsstat -m` shows `proto=rdma` and `config.json` is visible.
4. Build the patch chain v1→v8 over the day-0 arm64 image (patch files MUST be in the build-context root for the COPY steps), ship to all ranks.
5. Pre-launch on every node: start cache-flusher sidecar; `sync; echo 3 > /proc/sys/vm/drop_caches`.
6. Launch **worker-first** (TP4: ranks 3→2→1, ~22 s apart; head last). Health wait ~14–16 min (weights ~9 min + warmup ~3–5 min).
7. Verify: coherent greedy output, finite logprobs, `SpecDecoding metrics` lines if MTP enabled.

## Key configuration (what each critical flag does)

- `--block-size 2304` — kpool storage must tile by 64 for DeepGEMM arch-12 fp8 paged-MQA; 2304 = kpool·64 multiple and MLA 128-aligned.
- `--gpu-memory-utilization 0.85` — 0.78–0.80 starve the KV cache at long context.
- `--kv-cache-memory <vLLM's suggested number>` — REQUIRED with MTP (draft head +~5GB otherwise trips GB10 NVRM OOM). Use the startup-log value verbatim; larger values die (ladder study).
- `--kv-cache-dtype fp8_e4m3` — halves KV bytes; needs FlashInfer ≥0.6.18 + the smem tile-cap patch (v8).
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` — native MTP head; 3 beats 4 on prose (position-4 acceptance ~0.15–0.23 costs more than it returns).
- `--moe-backend marlin` — validated NVFP4 path on sm121.
- `--enforce-eager` — day-0 stability; CUDA graphs untested on this stack.
- NCCL fabric env: `NCCL_NET=IB NCCL_IB_HCA=rocep1s0f0 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_ADDR_RANGE=10.10.10.0/24`, socket/gloo/tp/mn on `enp1s0f0np0`, `NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0`. Do NOT set `NCCL_IB_GID_INDEX` (see TP4 section).
- Docker: `--network host --ipc host --shm-size 32g --ulimit memlock=-1 --cap-add IPC_LOCK --device /dev/infiniband`, model ro-mounted; TP4 adds `--memory 112g --memory-swap 112g`.

## Performance Reference (measured 2026-08-27, on-box, prose 512 tok, temp 0, thinking off, median of 3)

### TP2 (cb98 + 3b24) — autoresearch sweep

| Config | Decode tok/s | TTFT s | KV pool tokens | Notes |
|---|---:|---:|---:|---|
| bf16 KV, no MTP | 14.75 | 0.252 | 588,225 | reproduces tonyd2wild 14.3 |
| fp8 KV, no MTP | 14.65 | 0.229 | 1,020,772 | >1M pool on TP2 (no MTP head = +5GB headroom) |
| fp8 KV + MTP4 | 25.99 | 0.266 | 507,041 | their flagship; beats their 21.8 on our fleet |
| **fp8 KV + MTP3** | **28.29** | 0.338 | 527,879 | **winner**; draft accept 55–61%, mean accept len 2.65–2.83 |

### TP2 winner battery (fp8 KV + MTP3; C1/C4/C8, median of 3, warm pass, 512 tok)

| Cell | Aggregate tok/s | Per-stream tok/s | TTFT s |
|---|---:|---:|---:|
| Prose C1 | 27.73 | 28.20 | 0.342 |
| Prose C4 | 68.38 | 17.69 | 0.403 |
| Prose C8 | 73.40 | 15.55 | 0.425 |
| Code C1 | 28.08 | 28.59 | 0.360 |
| Code C4 | 65.63 | 17.20 | 0.433 |
| Code C8 | 69.27 | 14.77 | 0.735 |

C1 matches the sweep probe (28.2 vs 28.29 — harness consistency). C≥4 numbers carry ±10% warm-state variance; max-num-seqs=6 queues 2 requests at C8.

### TP4 (cb98 + 3b24 + 9105 + bdea) — autoresearch sweep

| Config | Decode tok/s | TTFT s | KV pool tokens | Notes |
|---|---:|---:|---:|---|
| fp8 KV + MTP4 (recipe flagship) | 38.18 | 0.223 | ~1.26M | beats tonyd2wild's 35.7 headline on our fleet |
| **fp8 KV + MTP3** | **38.63** | 0.216 | **1,292,571** (4.93x @262K) | **winner**; draft accept 50–57%, mean accept len 2.49–2.71 |
| bf16 KV, no MTP, unpinned | DNB | — | (50.37 GiB available) | died in KV alloc — unpinned ~50 GiB slab hits the GB10 NVRM fail-fast wall (KV-ladder mechanism at TP4 scale); pinned fp8 budget is the correct config |

Fleet-specific fix required vs the source recipe: `NCCL_IB_GID_INDEX` must be OMITTED on this cluster — bdea's IPv4 RoCE GID sits at idx 4–5 while the other nodes are at 2–3 (reboot-volatile GID tables); hardcoding idx 3 makes bdea pick a link-local GID → `ibv_modify_qp failed with 22` → all-rank rendezvous death. NCCL auto-detects correctly when unset (validated on 4 boots).

### TP4 winner battery (fp8 KV + MTP3; C1/C4/C8, median of 3, warm pass, 512 tok)

| Cell | Aggregate tok/s | Per-stream tok/s | TTFT s |
|---|---:|---:|---:|
| Prose C1 | 40.00 | 40.60 | 0.216 |
| Prose C4 | 109.96 | 28.28 | 0.285 |
| Prose C8 | 114.35 | 24.37 | 0.337 |
| Code C1 | 40.55 | 41.21 | 0.226 |
| Code C4 | 109.95 | 28.55 | 0.317 |
| Code C8 | 110.91 | 23.94 | 0.353 |

C1 40.0–40.6 vs sweep probe 38.63 (warm-state uplift, same harness). MTP3 acceptance under load: mean accept len 2.71–3.03, per-position ~0.83/0.59/0.42, draft accept 57–68%. vs tonyd2wild's TP4 flagship claim (35.7 tok/s): +12–14% single-stream, 3.2x at C8 aggregate. max-num-seqs=6 queues 2 requests at C8.

## Troubleshooting (symptom → cause → fix)

- Warmup dies `pe_dim must be 64 for fp8_ds_mla` → stock SM12x sparse backend requires DeepSeek packed cache; GLM is NoPE → patch 1 (SM90 NoPE backend → SM121).
- Serves but garbage/NaN output → FlashInfer 0.6.17 FA2 MLA NaN on 64–256-row batches → upgrade to 0.6.18 nightly (patch 2), then RE-PIN `nvidia-nccl-cu13==2.30.7` and `nvidia-cutlass-dsl==4.6.2` (the nightly silently downgrades both; 2.29.7 is fabric-fatal).
- `ncclCommInitRank: internal error` → NCCL downgrade (above).
- Rank dies silently post-load with MTP → GB10 NVRM allocation failure; pin `--kv-cache-memory` to vLLM's suggested value; run cache-flusher during boot; NFS-loading workers fail first.
- KV > suggested value → NVRM OOM on some rank every boot (ladder: 5.5/6.5/7.5 GiB pins all died at TP2+MTP).
- Boot flake after many cycles → reboot the node (driver allocation-pool degradation).
- Head node deep-wedges during unpinned-KV boot on local weights (kernel netstack alive — Tailscale online, TCP:22 accepts — but sshd can't fork): the unpinned ~50 GiB KV slab races page cache filled by the local weight read and exhausts the GB10 NVRM allocator. Only fix is a physical power cycle (~2–8 min to SSH). Always pin `--kv-cache-memory` on the head node; keep weights on NFS workers, not the head's local read path, for marginal configs.
- **`rpc.nfsd -r <port>` silently strands ~21 GiB of unified memory on the NFS head** (117.8→96.4 GiB MemFree, instant, reproducible; invisible to meminfo/slabinfo; survives nvidia module rmmod — it is NOT GPU memory). vLLM's startup check then dies: `Free memory ... 84.6 GiB ... less than desired GPU memory utilization (0.85, 103.4 GiB)`. The benign way to enable NFS-RDMA is ONLY `modprobe svcrdma; echo rdma 20049 > /proc/fs/nfsd/portlist` — verified holding 117.6 GiB with 3 live RDMA clients. Never put `rpc.nfsd -r` in the persist service; an nfs-server restart clears portlist anyway (re-echo after every restart). Diagnosis ladder when a node shows a pure MemFree deficit vs siblings: diff meminfo categories → stop nfs-server and re-measure → A/B the rdma enable method.
- Gateway/other hosts may not reach `<spark>:8000` over tailnet — bench on-box on the head node (localhost), which also removes the network hop from timing.

## Compared: other models on same hardware

(to be filled after Qwen3.8-Flash-Next NVFP4 campaign on the same pair)

## Credits

- Model: zai-org/GLM-5.3-Flash · Quant: LibertAIDAI/GLM-5.3-Flash-NVFP4
- Day-0 recipes: tonyd2wild/GLM-5.3-Flash-NVFP4-262K-2x-DGX-Spark and .../GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark (patch chain, flags, ops rules, KV-ladder study)
- Fleet: Vikas Sridhar's CRS812 cluster; campaign by Hermes Agent
