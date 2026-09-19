# Runbook: GLM-5.3-Flash NVFP4 + DFlash2 on 2× DGX Spark (TP2, 262K ctx)

- **Model:** zai-org/GLM-5.3-Flash (320B total / 18B active MoE) — quant: **RedHatAI/GLM-5.3-Flash-NVFP4** (compressed-tensors, W4A4, 185 GB, 11 shards — corruption-free; ModelOpt quants emit intermittent bad token IDs, vLLM #54150) + drafter **incoai/GLM-5.3-Flash-DFlash2** (2.2 GB)
- **Source recipe:** [tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) (CURRENT.md, 2026-09-18)
- **Hardware:** 2× DGX Spark (GB10/SM121) — head **aitopatom-cb98** (10.10.10.14, rank 0, serves :8000) + worker **gx10-141d** (10.10.10.16, rank 1, `--headless`). 200G CRS812 RoCEv2 fabric, rail `rocep1s0f0`/`enp1s0f0np0`. Weights LOCAL on each node (no NFS at boot).
- **Engine:** `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (digest `sha256:4def0ef644cb…`, verified on both nodes) + bind-mounted patches: SM121 top-k kpool fix, #18 prefix-cache repair (`kv_cache_coordinator.py`).
- **Campaign date:** 2026-09-19 · campaign dir `~/glm53-tp2-campaign/` on edgexpert-9105 · results `results/bench_<arm>.json`
- **Endpoint:** `http://10.10.10.14:8000/v1` (served name `glm-5.3-flash`)

## Delivered configuration (sweep winner — "k5g + prefixfix")

Shipped recipe with exactly two changes: **DFlash2 depth k=7→5** and **CUDA graphs on** (`--enforce-eager` dropped), plus the #18 prefix-cache repair bind-mount.

| flag | value | note |
|---|---|---|
| `--speculative-config` | `{"method":"dflash","model":"/models/dflash2-draft","num_speculative_tokens":5}` | k=5 (repo default 7) |
| `--enforce-eager` | **absent** | graphs on (repo keeps eager; refuted on our battery) |
| `--max-num-seqs` | 6 | mns8 helps only without graphs |
| `--kv-cache-memory` | 8589934592 (8 GiB) | pool 855,417 tok @ k5 (788,977 @ k7) |
| `--gpu-memory-utilization` | 0.85 | |
| `--max-model-len` | 262144 | do NOT raise on TP2 (KV starvation) |
| `--max-num-batched-tokens` | 8192 | 16384 → NVRM OOM (repo-verified) |
| `--block-size` | 2304 | DeepGEMM arch-12 fp8 paged-MQA |
| `--moe-backend` | marlin | |
| `--kv-cache-dtype` | fp8_e4m3 | |
| PREFIX_FIX | on (`~/patches/kv_cache_coordinator.py`) | agent-traffic TTFT −70% (repo) |
| docker | `--memory 112g --memory-swap 112g --ulimit memlock=-1 --cap-add IPC_LOCK --device /dev/infiniband --network host --ipc host --shm-size 32g` | |
| JIT storm cap | `MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1` + persistent `/var/tmp/glm53-vllm-cache/{tilelang,triton,flashinfer}` | |

NCCL fabric env (fleet-proven, NOT the repo's 192.168.192.x pins): `NCCL_NET=IB NCCL_IB_HCA=rocep1s0f0 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_ADDR_FAMILY=AF_INET NCCL_IB_ADDR_RANGE=10.10.10.0/24 NCCL_SOCKET_IFNAME=enp1s0f0np0` (gloo/tp/mn same), `NCCL_NVLS_ENABLE=0 NCCL_CROSS_NIC=0 NCCL_IB_MERGE_NICS=0 NCCL_CUMEM_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1`, **no GID pin** in NCCL env.

## Step-by-step

1. **Weights** (once per node): `RedHatAI/GLM-5.3-Flash-NVFP4` → `~/models/glm-5.3-flash-nvfp4-redhat` (185 GB, 21 files, must include `chat_template_mm.jinja`), drafter → `~/models/GLM-5.3-Flash-DFlash2`. Node-to-node: 4 parallel per-shard rsync streams over fabric ≈ 9 min for 185 GB (single stream caps ~370 MB/s). Verify: per-file `stat -c "%n %s"` diff; `config.json` `quant_method` must be `compressed-tensors`.
2. **Image** on both nodes: `docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`; verify digest `sha256:4def0ef644cb…`.
3. **Patches** on both nodes: `cp <repo>/docker/sparse_attn_indexer_kpool_sm121.py ~/patches/sparse_attn_indexer_kpool.py`. PREFIX_FIX: run `<repo>/docker/dflash2-overlay/patch_prefix_cache_draft_group.py` inside a throwaway container of the image (it patches `/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py` in place; self-check must print OK), copy result to `~/patches/kv_cache_coordinator.py` (keep `.orig`).
4. **Launcher:** `<repo>/launch-glm53-vllm-tp2-dflash2.sh` adapted — head/worker IPs 10.10.10.14/.16, `MODEL_HOST_PATH`/drafter paths per node, NCCL env above, sweep knobs via env (`MNS MNBT GMU KV_MEM SPEC_JSON EAGER_ON PREFIX_FIX ROCE`). ⚠️ Do NOT use `VAR="${VAR:-{...json...}}"` defaults — brace-in-default corrupts env-provided JSON; use `if [ -z … ]`.
5. **Pre-launch:** GPU clock-latch burn check both nodes (15 s fp16 8192³; healthy ≈ 93–95 TFLOPS, latched ≈ ⅓ → 30–60 s power unplug). Then drop caches — no passwordless sudo on these nodes, use: `docker run --rm --privileged -v /proc/sys:/hostsys --entrypoint bash <image> -c 'sync; echo 3 > /hostsys/vm/drop_caches'`. Start gentle flusher (drop only when MemAvailable < 8 GiB).
6. **Launch worker FIRST** (`launch-tp2.sh 1` on gx10-141d), sleep 25, then head (`launch-tp2.sh 0` on cb98). Readiness ~8–13 min warm-JIT (~15 min cold). Poll `/health` — **never `/v1/models`** (200 with dead engine).
7. **Correctness gate:** Hangul probe 3 prompts × 3 passes temp 0 → U+FFFD + control chars must be 0; forced tool-call with typed enum args; streaming sanity.
8. **Bench:** C1 prose + C1 code (median of 3) + C4/C8 aggregate (median of 3), 256 out tokens, temp 0, thinking off. Quote prompt class with every number — spec-decode throughput is acceptance-bound.

## Performance (measured 2026-09-19, battery above, median tok/s)

| config | C1 prose | C1 code | C4 agg | C8 agg |
|---|---|---|---|---|
| shipped recipe (k7, eager) | 18.80 | 29.42 | 40.83 | 59.78 |
| k5 + graphs (arm k5g) | 22.92 | 32.48 | 46.97 | 69.98 |
| **delivered (k5 + graphs + prefixfix)** | **22.73** | **32.09** | **51.20** | **72.40** |
| best C4 variant (k5+mns8+graphs) | 21.46 | 30.21 | 49.53 | 69.03 |

**Δ delivered vs shipped: C1 prose +20.9%, C1 code +9.1%, C4 +25.4%, C8 +21.1%.** The delivered C4/C8 edge over the k5g arm is PREFIX_FIX on repeat-prompt traffic (prefill savings; battery reuses prompts) — decode-only expectation is the k5g row. PREFIX_FIX verified live: repeated 6,755-token prompt → 4,608 cached tokens (2× block-2304 aligned), e2e 7.07 s → 2.68 s; note short prompts (< 2304 tokens) can never register a hit at this block size — hits=0 there is expected, not a fault.

Correctness gate (delivered engine): Hangul probe 3×3 temp 0 → 0 U+FFFD / 0 control chars; forced tool-call with enum args correct; streaming OK.

Sweep arms (all vs shipped baseline, fresh boot each): mns8 (+2.4%/+5.9% C4/C8) · k5 (+10.0%/+18.1%) · k5sched `[[1,3,7],[4,512,5]]` (C4 +11.5%, C8 +12.6%) · graphs (+7.9%/+7.1%) · prefixfix (+4.8%/+9.4%, prefill-driven) · k5m8 · k5g · k5m8g. Repo claims refuted on our battery: k7-better-single-stream and graphs-flat-at-TP2 are prompt-class-relative; k5+graphs stack super-linearly at C1.

## Untested / blocked

- **b12x RoCEnante RoCE all-reduce** (repo: +5–18% agg): needs `vllm-dsv41:exl3b-roce` runtime (author-local image); no fleet image carries the full b12x comm/roce set. Not refuted — untestable until the bundle source is available.
- 262K-context depth ladder and soak not run this campaign (battery was short-prompt decode-focused).

## Troubleshooting

- **Worker container exits at boot with `argument --speculative-config: Value {...}} cannot be converted`** — brace-in-default launcher bug (see step 4).
- **`CONTAINER-GONE` mid-boot** — check `docker logs vllm_glm53` on both nodes; NVRM OOM → mnbt back to 8192 / KV pin down.
- **Sudden ~⅓ speed on both ranks** — GB10 clock latch; burn-check, power-unplug 30–60 s.
- **Engine dead but `/v1/models` returns 200** — always gate on `/health` + a real completion.
- Final teardown is mandatory: `docker rm -f vllm_glm53` both nodes, stop flushers (`touch /tmp/flusher_stop`), drop caches.
