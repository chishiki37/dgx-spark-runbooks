# Runbook: MiMo-V2.6-Flash-RL on 2× DGX Spark (TP2, DFlash-7, 300K ctx, omnimodal)

- **Model:** `XiaomiMiMo/MiMo-V2.6-Flash-RL` (309B total / 15B active sparse MoE, native FP8 e4m3 dynamic + MXFP4 experts, 1M native context, text+image+video+audio) — **177.7 GB on disk**: 64 expert-parallel shards (`model_pp0_ep<N>_shard0.safetensors`) + `model_mtp.safetensors` + `dflash/dflash_draft_model.safetensors` (5-layer block drafter, `block_size 8`) + `audio_tokenizer/model.safetensors`
- **Source recipe:** [tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark](https://github.com/tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark) (README status 2026-09-22; four patched vLLM files shipped as bind mounts)
- **Hardware:** 2× DGX Spark (GB10/SM121, 121.7 GiB unified each) — head **edgexpert-04af** (10.10.10.15, rank 0, serves :8888, weights LOCAL) + worker **edgexpert-3b24** (10.10.10.12, rank 1 `--headless`, weights via **Docker local-driver NFS volume** from the head — zero worker sudo). CRS812 RoCEv2 fabric, rail `rocep1s0f0`/`enp1s0f0np0`, `ADDR_RANGE=10.10.10.0/24`.
- **Engine:** `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (vLLM 0.1.dev20051+g487ecf187, torch 2.13, CUDA 13, sm_120 kernels) + repo patches bind-mounted: fused-FP8-QKV TP2 reshard (`mimo_v2.py`), `SupportsEagle3` marker on the omni class (`mimo_v2_omni.py`), fp8-KV DiffKV backend (`triton_attn_diffkv.py`), optional drafter value-scale (`qwen3_dflash.py`, off). Plus fixed `dflash/config.json` (upstream has a trailing comma) and `soundfile`+`PyAV` staged to `$CACHE/pyextra` for audio input.
- **Campaign date:** 2026-09-22 · campaign dir `~/mimo-campaign/` on edgexpert-04af · raw results `{baseline,seqs16,spectok8,final-baseline}/bench-*.json`
- **Endpoint:** `http://10.10.10.15:8888/v1` (served name `mimo-v2.6-flash`)

## Delivered configuration (sweep winner = recipe defaults)

Our two-knob sweep could not beat the upstream defaults — delivered config is the repo's, which makes this a reproduction-plus-C8-data-point runbook.

| flag | value | note |
|---|---|---|
| `--tensor-parallel-size` | 2 | `--nnodes 2`, mp backend, rank 1 `--headless` |
| `--speculative-config` | `{"method":"dflash","model":"/models/mimo/dflash","num_speculative_tokens":7}` | k=8 refuted (below); drafter `block_size 8` is the structural cap |
| `--kv-cache-dtype` | fp8 | pool **1,865,221 tokens** (6.22× at 300K) |
| `--gpu-memory-utilization` | 0.90 | |
| `--max-model-len` | 300000 | 1M not attempted (per-request blocks ≈ 3.3×) |
| `--max-num-seqs` | 8 | 16 refuted (below) |
| `--moe-backend` | marlin | MXFP4 experts; **DeepGEMM must stay off** (`VLLM_USE_DEEP_GEMM=0`, silent SM12x fp8 corruption, DeepGEMM#417) |
| `--default-chat-template-kwargs` | `{"enable_thinking": false}` | else reasoning leaks into `content` |
| parsers | `--reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice` | |
| docker | `--memory 112g --memory-swap 112g --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband --network host --ipc host --shm-size 32g` | |

NCCL fabric env (repo defaults matched our fleet): `NCCL_NET=IB NCCL_IB_HCA=rocep1s0f0 NCCL_IB_GID_INDEX=3 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_ADDR_FAMILY=AF_INET NCCL_IB_ADDR_RANGE=10.10.10.0/24 NCCL_SOCKET_IFNAME=enp1s0f0np0` (gloo same), `NCCL_NVLS_ENABLE=0 NCCL_CROSS_NIC=0 NCCL_IB_MERGE_NICS=0 NCCL_CUMEM_ENABLE=0 NCCL_IGNORE_CPU_AFFINITY=1`.

## Step-by-step

1. **Weights on the head only** (resumable, retry-looped stager; `.incomplete` resume works): `snapshot_download("XiaomiMiMo/MiMo-V2.6-Flash-RL", local_dir=~/models/MiMo-V2.6-Flash-RL, max_workers=16)`. Measured 81 MB/s WAN → ~2 h for 177.7 GB. Verify: **64/64** `model_pp0_ep*_shard0.safetensors` + `model_mtp.safetensors` + `model.safetensors.index.json` + `config.json` + `dflash/` + `audio_tokenizer/` (the standard `model-*-of-*.safetensors` glob does NOT match this EP layout).
2. **Image** on both nodes: `docker pull ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (~63 GB; retry-loop it if the WAN is saturated — DNS timeouts happen mid-download campaigns).
3. **Head setup:** `SKIP_DOWNLOAD=1 bash setup.sh` (stages the four patched files + fixed dflash config to `/var/tmp/mimo-cache`, installs `soundfile`/`av` to `$CACHE/pyextra`).
4. **NFS (head, once):** append to `/etc/exports`: `<model_dir> 10.10.10.0/24(ro,sync,no_subtree_check,no_root_squash) 10.10.20.0/24(…)`, `exportfs -ra`. RDMA port already live fleet-wide (`rdma 20049` in `/proc/fs/nfsd/portlist`).
5. **Worker volume (zero sudo):** `docker volume create --driver local --opt type=nfs --opt device=10.10.10.15:<model_dir> --opt o=ro,addr=10.10.10.15,nfsvers=3,tcp,rsize=1048576,wsize=1048576,timeo=600,retrans=2 mimo-flash-nfs`. Probe before launch: `docker run --rm --mount type=volume,source=mimo-flash-nfs,destination=/m,readonly --entrypoint ls <image> /m/config.json`. Stage the same `$CACHE` contents as the head (patch files are plain copies from the repo).
6. **Launch worker FIRST** (`serve.sh 1` variant using `--mount type=volume,…,readonly` in place of the host bind mount), then head (`serve.sh 0`). Load ≈ **11 min** (worker reads its half over NFS-TCP; vLLM detects NFS and skips auto-prefetch — checkpoint > 90% of RAM). Ready when `/v1/models` lists `mimo-v2.6-flash`; first boot adds ~2 min torch.compile (warm cache after).
7. **Correctness gate:** greedy exact-reply probe (thinking off → `reasoning_chars 0`), math one-liner (17×23=391 ✓), vision: generated red-square/blue-circle/green-triangle/"MIMO 42" card → all shapes, colors, positions and text read correctly (2.6 s round trip). Audio/video: upstream-verified; not re-run here.
8. **Bench:** `bench/mimobench.py --levels 1,4,8` (prompt set v1, temp 0, thinking off, salted unique prefixes → no prefix-cache inflation; tokens from server `usage` — spec decode packs multiple tokens per chunk). Quote category with every number: DFlash throughput is acceptance-bound (counting 6.9/7 vs prose ~1.2/7).

## Performance (measured 2026-09-22, C1+C4+C8 battery, two boots of the delivered config)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| 1 | 46.2–46.8 | **54.3–55.0** | 0.36–0.38 |
| 4 | **117.6–121.4** | 34.9–35.9 | 0.43–0.45 |
| 8 | **190.1–193.9** | 28.2–29.0 | 0.54–0.55 |

Run-to-run variance ±3% (two independent boots). Per-stream C1 spread by category (first boot): structured 86.5 · ceiling-count 90.7 · math 74.4 · coding 70.6 · json 53.4 · reasoning 41.8 · summary 31.1 · prose 26.4 · narrative 23.4.

Cold prefill (unique prefix): **2,044 tok/s @2K** (TTFT 0.98 s) · 1,754 @8K · 1,484 @32K · **1,270 @64K** (TTFT 50 s). Warm-compile boot measured up to 2,320 @2K / 1,929 @32K.

DFlash acceptance (of 7) at C1: ceiling-count 6.8–6.9 · structured 6.4–6.9 · format 6.0–6.2 · coding 5.1–5.2 · math 5.0 · json 3.6–4.0 · reasoning 2.5–2.6 · summary 1.6–1.8 · prose 1.2 · narrative 0.8.

### Vs upstream published (their pair B, fp8 KV, same config; their table stops at C6)

| C | upstream | ours | Δ |
|---|---|---|---|
| 1 (per-stream) | 53.31 | 54.3 / 55.0 | +2–3% |
| 4 (aggregate) | 120.37 | 121.4 / 117.6 | parity (±3%) |
| 6 (aggregate) | 155.77 | — | |
| 8 (aggregate) | — | **190.1 / 193.9** | new data point, +22–24% over their C6 |

Full reproduction of the upstream recipe on different hardware/fabric; no regression found.

## Sweep arms (both refuted; fresh boot + full battery each)

| arm | C1 per-stream | C4 agg | C8 agg | verdict |
|---|---|---|---|---|
| **delivered (SEQS 8, k7)** | **54.3 / 55.0** | **121.4 / 117.6** | **193.9 / 190.1** | winner |
| SEQS=16 | 55.1 (+1.5%, noise) | 103.9 (**−14%**) | 174.2 (**−10%**) | rejected |
| DFlash k=8 | 51.7 (−4.8%) | 113.6 (−6.4%) | 181.8 (−6.2%) | rejected |

k=8 detail: the saturated categories did improve (ceiling-count C1 90.7→96.1, C4 62.8→71.0; structured C1 86.5→89.4) but every mid/low-acceptance category paid the extra draft+verify cost — net negative on any mixed workload. `block_size 8` means k=8 is the structural maximum; no further depth exists.

Skipped on prior evidence: `DFLASH_VSCALE=1` (upstream measured zero effect), DeepGEMM MoE (SM12x corruption), GMU>0.90 (no decode gain at 6.2× KV oversubscription; startup-refusal risk), 1M context (per-request blocks ≈3.3× of 300K).

## Untested / blocked

- Audio + video input paths (upstream-verified; libs staged here but not exercised).
- 1M-context depth ladder; soak.
- Alternative MXFP4 MoE backends (marlin is the only SM12x-safe path in this image).
- TP4 / PP topologies for Flash; Pro-RL (1.02T/42B, 573 GB FP8) serve — weights staging only at this time.

## Troubleshooting (appendix)

- **Stager hangs with zero disk writes, single thread, one open socket** — `snapshot_download` metadata stall (observed on the Pro-RL repo after a long Flash campaign; curl to the same API works). Kill + relaunch; `HF_HUB_DISABLE_XET=1` did not change it — root cause still open; the retry loop only catches exceptions, not hangs, so add an external liveness check (du growth over 2 min) to unattended stagings.
- **Image pull DNS timeout (`lookup ghcr.io … i/o timeout`)** while a model download saturates the WAN — retry-loop the pull; it succeeds between bursts.
- `tests/mimo_test.py` takes the **full** endpoint URL (`…/v1/chat/completions`), not the base — 404 otherwise.
- **Head restart at GMU 0.90**: vLLM probes free memory and refuses below ~111 GiB available; `serve.sh` waits, but pre-dropping caches (`sync; echo 3 > /proc/sys/vm/drop_caches` via sudo) makes it deterministic. Worker passes without sudo (NFS page cache counts as reclaimable).
- Bench chaining: poll `/v1/models` (this image's engine and API come up together; the `/health`-only pitfall from the GLM-5.3 lane was not observed) then run the battery — ~11 min load + ~7 min bench per arm.
- **Do not pause-resume the Pro-RL WAN download mid-bench** on the head node: 16-worker disk writes add prefill noise (observed +13–30% prefill swings between boots are compile-cache warmth, but keep downloads paused during decode arms for clean numbers).
