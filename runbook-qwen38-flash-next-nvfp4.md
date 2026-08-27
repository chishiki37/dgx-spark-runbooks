# Runbook: Qwen3.8-Flash-Next NVFP4 on 2× DGX Spark (TP2, SGLang)

- **Model:** Qwen3.8-Flash-Next (`qwen4_exp` arch — hybrid linear-attn/mamba + sparse attention, NEXTN speculative head, 51B-parameter PLE embedding table, vision-capable, 262K ctx) — quant: RadixArk/Qwen3.8-Flash-Next-NVFP4 (135.3 GB)
- **Hardware:** NVIDIA DGX Spark (GB10, SM121) ×2 — aitopatom-cb98 (head) + edgexpert-3b24 (worker). 200G CRS812 RoCEv2 fabric, NFSoRDMA weights from cb98.
- **Engine:** SGLang day-0 image `lmsysorg/sglang:qwen38flashnext` + 2-patch SM121 chain → local tag `sglang-qwen38fn:sm121-qsa` (QSA guard + NCCL pin; see Troubleshooting).
- **Performance (measured, on-box, temp 0, thinking off):** see Performance Reference below.
- **Endpoint:** `http://<head>:8000/v1` (served name `qwen3.8-flash-next`)
- **Campaign date:** 2026-08-27

## Prerequisites

1. Two DGX Sparks with fabric links UP on rail 1 (`ip -br addr` shows 10.10.10.x UP; `ibdev2netdev` shows `rocep1s0f0 ==> enp1s0f0np0 (Up)`).
2. Weights staged on the head and NFS-exported read-only to both fabric subnets; worker mounts over NFSoRDMA (vers=3, proto=rdma, port 20049). **Enable RDMA with `modprobe svcrdma; echo rdma 20049 > /proc/fs/nfsd/portlist` ONLY — never `rpc.nfsd -r`** (it strands ~21 GiB of unified memory invisibly; see GLM runbook troubleshooting).
3. Patched image on both ranks (`docker save | ssh <node> docker load` over fabric). Verify `grep '^IMAGE'` matches before every launch.
4. sudo on both nodes for `drop_caches` + cache-flusher sidecar (GB10 NVRM allocates KV from MemFree only).

## Step-by-step

1. Download `RadixArk/Qwen3.8-Flash-Next-NVFP4` (135.3 GB) to the head node.
2. NFS-export the models dir (ro,no_root_squash) to `10.10.10.0/24` + `10.10.20.0/24`; enable `rdma 20049` in portlist (re-echo after every nfs-server restart).
3. Worker: mount `vers=3,proto=rdma,port=20049`, verify `config.json` visible.
4. Build the patched image over the day-0 arm64 image (QSA guard patch + `pip install nvidia-nccl-cu13==2.30.7`), ship to both ranks.
5. Pre-launch on both nodes: cache-flusher sidecar + `sync; echo 3 > /proc/sys/vm/drop_caches`.
6. Launch **worker first** (rank 1), head ~25 s later. Health wait ~8 min (weights load + CUDA graph capture).
7. Verify: `/v1/models` lists `qwen3.8-flash-next`, coherent greedy output.

## Key configuration (what each critical flag does)

- `--quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass` — the validated NVFP4 GEMM path on SM121.
- `--page-size 64` — KV page size matched to the sparse-attn backend.
- `--mamba-scheduler-strategy extra_buffer --mamba-track-interval 64` — hybrid mamba/linear-attn state scheduling (SSM checkpoint kept bit-identical via `--enable-linear-replayssm-spec`, which forces mamba ssm dtype float32).
- `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` — native NEXTN head; sweep confirms 3/4 beats 2/3 (46.7 vs 46.1 tok/s) — unlike GLM's MTP3>MTP4 result, the recipe default is already optimal here.
- `--chunked-prefill-size 4096 --max-running-requests 6` — prefill chunking + concurrency cap (C8 queues 2).
- `--context-length 262144 --max-total-tokens 600000 --mem-fraction-static 0.80` — production KV pin: 600K tokens (bf16 KV, ~6.9 GB/rank K+V). 0.82 buys nothing measurable (46.3 vs 46.7) and higher fractions OOM under load per the source recipe's KV-ladder study (0.82 + offloads reaches 1.05M tokens but dies under load).
- `--allow-auto-truncate --ple-offload-embedding` — PLE embedding table (51B params) offloaded to host RAM; truncation safety for over-long inputs.
- `--cuda-graph-max-bs 8 --disable-cuda-graph-padding` — decode graphs up to batch 8.
- `--disable-radix-cache --sampling-backend pytorch --default-chat-template-kwargs '{"enable_thinking": false}'` — agent-safety stack: prevents the token-0 `!` loop at temp ≤0.7 (sglang #36537 class). Keep agent temps ≤0.7.
- `--reasoning-parser auto --tool-call-parser qwen3_coder` — chat/tool plumbing.
- NCCL fabric env: `NCCL_NET=IB NCCL_IB_HCA=rocep1s0f0 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_ADDR_RANGE=10.10.10.0/24 NCCL_MAX_NCHANNELS=4 NCCL_MIN_NCHANNELS=4 NCCL_CROSS_NIC=1`, socket/gloo/tp/mn on `enp1s0f0np0`, `NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0`. Do NOT set `NCCL_IB_GID_INDEX` (reboot-volatile GID tables differ across nodes — see GLM runbook).
- Docker: `--network host --ipc host --shm-size 32g --memory 110g --memory-swap 110g --ulimit memlock=-1 --cap-add IPC_LOCK --device /dev/infiniband`, model ro-mounted.

## Performance Reference (measured 2026-08-27, on-box head, 512 tok out, temp 0, thinking off, median of 3)

### TP2 autoresearch sweep (prose 512 tok probe)

| Config | Decode tok/s | TTFT s | Notes |
|---|---:|---:|---|
| **q0: recipe-locked (memfrac 0.80, NEXTN 3/4, 600K)** | **46.74** | 0.165 | **winner**; matches source "47 typical" claim |
| q1: NEXTN steps 2 / draft 3 | 46.14 | 0.158 | lighter spec loses ~1.3% |
| q2: memfrac 0.82 | 46.31 | 0.167 | no gain, KV stays pinned 600K |

KV pool: 600,000 tokens (bf16, K 3.43 GB + V 3.43 GB per rank) at all arms.

### TP2 winner battery (q0; C1/C4/C8, warm pass, 512 tok)

| Cell | Aggregate tok/s | Per-stream tok/s | TTFT s |
|---|---:|---:|---:|
| Prose C1 | 48.09 | 48.76 | 0.166 |
| Prose C4 | 119.71 | 31.11 | 0.191 |
| Prose C8 | 126.09 | 29.71 | 0.196 |
| Code C1 | 63.82 | 65.10 | 0.172 |
| Code C4 | 163.03 | 44.45 | 0.197 |
| Code C8 | 159.93 | 34.09 | 0.211 |

Code content decodes ~33% faster than prose (NEXTN acceptance is content-driven — structured tokens verify better): C1 code 63.8 tok/s lands inside the source recipe's "53–64 structured / 70 peak" claim band; prose C1 48.1 matches "47 typical".

## Compared: GLM-5.3-Flash NVFP4 on the same pair (cb98+3b24, TP2, same harness)

| Cell | Qwen3.8-FN (TP2) | GLM-5.3 (TP2) | GLM-5.3 (TP4) |
|---|---:|---:|---:|
| Prose C1 | 48.1 | 27.7 | 40.0 |
| Prose C8 | 126.1 | 73.4 | 114.4 |
| Code C1 | 63.8 | 28.1 | 40.6 |
| Code C8 | 159.9 | 69.3 | 110.9 |

Qwen3.8-Flash-Next on 2 nodes beats GLM-5.3-Flash on 4 nodes in every cell (SGLang CUDA graphs + NEXTN vs vLLM enforce-eager + MTP; Qwen's smaller active-param footprint). GLM counters with the larger KV pool (1.29M fp8 tokens at TP4 vs 600K bf16) and 262K-per-request headroom.

## Troubleshooting (symptom → cause → fix)

- Warmup dies `MLIRError: coord and shape weakly congruent` → SIGQUIT → QSA guard: the FlashInfer TRT-LLM sparse-decode kernel is gated behind `is_sm100_supported()`; SM121 falls back to the FA4 CUTE path which dies. One-line fix in `qwen_sparse_attn_backend.py`: `if not (is_sm100_supported() or is_sm120_supported()): return None` (import `is_sm120_supported` from `sglang.srt.utils`), then delete the stale `__pycache__/qwen_sparse_attn_backend.cpython-312.pyc`.
- `ncclCommInitRank: internal error` / fabric death at rendezvous → image bundles `nvidia-nccl-cu13 2.29.7` (fabric-fatal on Spark IB) → `pip install nvidia-nccl-cu13==2.30.7` in the image.
- Head shows ~21 GiB less MemFree than the worker → NFS head trap: `rpc.nfsd -r` strands unified memory; enable RDMA via portlist echo only (GLM runbook has the full diagnosis ladder).
- Token-0 `!` loop in agent/tool sessions → keep thinking-off + radix-off + pytorch-sampling stack and temp ≤0.7.
- Rare multimodal-rope device assert under CUDA graphs (~1/90 min, source report) → `--disable-cuda-graph` fallback (~55 peak, vision preserved).
- Boot dies silently / KV allocation fails → GB10 NVRM allocates from MemFree only; run the cache-flusher sidecar, drop_caches before launch, keep the 600K pin (the 1.05M-token 0.82 config OOMs under load).
- Gateway can't reach `<spark>:8000` over tailnet — bench on-box on the head (localhost).

## Credits

- Model: Qwen/Qwen3.8-Flash-Next · Quant: RadixArk/Qwen3.8-Flash-Next-NVFP4
- Day-0 recipe: tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark (locked flags, KV ladder, agent-safety stack)
- Fleet: Vikas Sridhar's CRS812 cluster; campaign by Hermes Agent
