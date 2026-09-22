# Runbook: MiMo-V2.6-Pro-RL on 8× DGX Spark (TP8, DFlash-7, Triton FP8 linear)

- **Model:** `XiaomiMiMo/MiMo-V2.6-Pro-RL` (native FP8 block-quantized MoE, omnimodal, 70 layers, hidden 6144, 128 attn heads / 8 KV heads, dense intermediate 16384, MoE intermediate 2048) — **573.5 GB on disk**: 130 safetensors shards + `dflash/` 5-layer drafter (5.5 GB, unquantized) + `audio_tokenizer/`.
- **Source recipe:** [tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark](https://github.com/tonyd2wild/MiMo-V2.6-Flash-2x-DGX-Spark) launch kit (`serve_pro.sh`, per-rank docker + Docker-NFS worker volumes), extended by this campaign to 8 ranks and a Triton linear backend.
- **Hardware:** 8× DGX Spark (GB10/SM121, 121.7 GiB unified each). Ranks: 0 = **edgexpert-04af** (10.10.10.15, head, serves :8888, weights LOCAL + NFS server) · 1 = 3b24 (.12) · 2 = ae1e (.13) · 3 = 9105 (.1) · 4 = bdea (.2) · 5 = cb98 (.14) · 6 = gx10-141d (.16) · 7 = 1d49 (.11, ComfyUI paused for the run). Workers mount weights via Docker local-driver NFS volume — zero worker sudo, no local copies. CRS812 RoCEv2 fabric, rail `rocep1s0f0`/`enp1s0f0np0`, `ADDR_RANGE=10.10.10.0/24`.
- **Engine:** `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` (vLLM 0.1.dev20051+g487ecf187, torch 2.13, CUDA 13) + bind-mounted patches: `mimo_v2.py`, `mimo_v2_omni.py`, `triton_attn_diffkv.py` (fp8 KV), `pyextra` (soundfile/PyAV audio).
- **Campaign date:** 2026-09-22 · campaign dir `~/mimo-campaign/` on edgexpert-04af · raw results `pro-baseline/bench-pro-baseline.{json,md}` (PP3×TP2 arm) and `pro-dflash7-tp8/bench-pro-dflash7-tp8.{json,md}` (delivered arm).
- **Endpoint:** `http://10.10.10.15:8888/v1` (served name `mimo-v2.6-pro`)

## Why TP8×PP1 (and not the 6-node PP3×TP2 that fits without borrowing nodes)

Pro-RL cannot share a node (573.5 GB / 121.7 GiB usable), so the topology must shard across ≥6 nodes. Three candidates:

| topology | weights/node | DFlash speculation | verdict |
|---|---|---|---|
| PP3×TP2 (6 nodes) | 95.6 GB | **blocked** — `DFlashDraftModel` does not implement `SupportsPP`; vLLM rejects any PP>1 with speculative config | runs spec-free only |
| TP4 (PP2) | 143.4 GB | — | doesn't fit (>121.7 GiB) |
| **TP8×PP1 (8 nodes)** | **71.7 GB** | **works** — PP1, all layers sharded 8-way | **delivered** |

TP8 needs `num_attention_heads % 8 == 0` (128 ✓), `num_kv_heads % 8 == 0` (8 ✓), `moe_intermediate % 8 == 0` (2048 ✓). TP6 fails all three — the model is TP8-or-bust for PP1.

## Delivered configuration

| flag | value | note |
|---|---|---|
| `--tensor-parallel-size 8 --pipeline-parallel-size 1` | 8 ranks, 8 nodes | mp backend, ranks 1–7 `--headless` |
| `--speculative-config` | `{"method":"dflash","model":"/models/mimo/dflash","num_speculative_tokens":7}` | 5-layer drafter, unquantized |
| `--linear-backend` | **triton** | **required**: CUTLASS block-FP8 kernel rejects this model's TP8 shard shapes at runtime (appendix A3); MoE experts unaffected (marlin) |
| `--moe-backend` | marlin | DeepGEMM stays off (`VLLM_USE_DEEP_GEMM=0`, SM12x fp8 corruption) |
| `--kv-cache-dtype` | fp8 | pool **4,457,547 tokens** (136× at 32K) |
| `--gpu-memory-utilization` | 0.80 | fits even with ~21 GiB orphaned on the head (appendix A6); KV still 19.24 GiB/rank |
| `--max-model-len` | 32768 | battery ceiling; KV pool supports far more (131K × ~34 concurrent) |
| `--max-num-seqs` | 8 | |
| docker | `--ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576` + standard fabric flags | **nofile is mandatory for TP8 mesh** (appendix A2) |
| correctness guards | `--no-async-scheduling`, `--generation-config auto`, `repetition_penalty 1.05`, thinking off | inherited from Flash 6651626 |

NCCL fabric env identical to the Flash runbook (`NCCL_NET=IB NCCL_IB_HCA=rocep1s0f0 NCCL_IB_GID_INDEX=3 NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_ADDR_FAMILY=AF_INET NCCL_IB_ADDR_RANGE=10.10.10.0/24`, NVLS/cross-NIC/merge-NICs off, CUMEM off).

## Performance (measured 2026-09-22, prompt set v1, temp 0, thinking off, salted unique prefixes)

### Delivered: TP8 + DFlash-7 + Triton linear

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| 1 | 30.19 | **35.23** | 0.54 |
| 4 | 54.18 | 16.47 | 1.12 |
| 8 | **71.43** | 10.70 | 1.60 |

Per-stream C1 by category: coding **55.9** · structured 54.1 · ceiling-count 54.1 · format 49.4 · math 40.5 · json 40.5 · reasoning 21.4 · narrative 19.8 · summary 18.3 · prose 17.2.

DFlash acceptance (of 7 drafted) at C1: ceiling-count **6.94** · structured 6.69 · format 5.88 · coding 5.19 · math 4.58 · json 2.79 · reasoning 2.03 · summary 1.62 · prose 1.20 · narrative 0.81. Acceptance is category-bound — always quote the category with the number.

Cold prefill (unique prefix): 463 tok/s @2K (TTFT 4.3 s) · 570 @8K (13.9 s) · 584 @32K (54.5 s).

### Vs the 6-node PP3×TP2 baseline (same battery, same day, spec-free)

| metric | PP3×TP2 (6 nodes, no spec) | TP8 + DFlash-7 (8 nodes) | gain |
|---|---|---|---|
| C1 aggregate | 9.37 | 30.19 | **3.2×** |
| C1 per-stream | 9.82 | 35.23 | **3.6×** |
| C4 aggregate | 25.10 | 54.18 | 2.2× |
| C8 aggregate | 37.62 | 71.43 | 1.9× |
| C8 TTFT | 3.03 s | 1.60 s | −47% |
| prefill @2K | 342 tok/s | 463 tok/s | +35% |
| prefill @32K | 724 tok/s | 584 tok/s | **−19%** (only regression; per-layer all-reduce vs stage-local compute — matters only for long-prompt batch jobs) |
| KV pool | 686,863 tok (21× at 32K) | 4,457,547 tok (136× at 32K) | 6.5× |

The gain compounds three effects: no pipeline bubbles (PP3 → PP1), speculation (up to 7 accepted tokens per verified step on high-acceptance categories), and per-node weights dropping 95.6 → 71.7 GB (KV 14.6 → 19.2 GiB/rank). PP3×TP2 C8/C1 aggregate scaling was only 4.0× — decode was bandwidth/bubble-bound, exactly the profile speculation amortizes.

## Step-by-step (deltas from the Flash runbook)

1. Stage weights on the head only (`stage_mimo.py`, expect 130 shards + `dflash/` + `audio_tokenizer/`); NFS-export the model dir to `10.10.10.0/24`; workers create the Docker NFS volume (probe with the throwaway `ls /m/config.json` container before launch).
2. Stage `/var/tmp/mimo-pro-cache` on **every** node: the three patched `.py` files + `pyextra/` (`triton/` and `huggingface/` subdirs are root-owned runtime artifacts — regenerated, don't rsync them).
3. Add `--ulimit nofile=1048576:1048576` to the docker run (both new nodes and old).
4. `serve_pro.sh <rank>` per node, head first; env `TP=8 PP=1 NNODES=8 SPEC=dflash GMU=0.80 MAXLEN=32768 EXTRA_ARGS="--linear-backend triton"`.
5. **First boot budget: ~45–60 min** (Triton JIT-compiles FP8 kernels for every hybrid-layer shape; cache persists to `/var/tmp/mimo-pro-cache/triton` → later boots ~15 min). Wait on `/v1/models`, and treat `docker inspect` state as the liveness signal — don't time out at 40.
6. Bench: `bench/mimobench.py --model mimo-v2.6-pro --levels 1,4,8 --prefill 2000,8000,32000`.

## Sweep ledger

| arm | result | verdict |
|---|---|---|
| PP3×TP2, GMU 0.90, 131K ctx | admission failure (needs 15.16 GiB reserve, 13.61 free after 95.6 GB weights) | rejected |
| PP3×TP2, GMU 0.85, 131K ctx | KV 5.0 GiB/rank — "No available memory for cache blocks" (one 131K request doesn't fit) | rejected |
| PP3×TP2, GMU 0.88, 32K ctx | boots; baseline banked (9.4/25.1/37.6) | reference |
| PP3×TP2 + DFlash | `NotImplementedError`: drafter lacks `SupportsPP` | structurally blocked |
| TP8 + DFlash, GMU 0.88, cutlass linear | NCCL fd exhaustion → after fix: CUTLASS "Invalid status" on N=3392 shard | rejected → fixed by A2+A3 |
| TP8 no-spec, cutlass linear | same CUTLASS failure (proves it's sharding, not speculation) | diagnostic arm |
| **TP8 + DFlash-7 + triton linear, GMU 0.80** | **30.2/54.2/71.4 agg** | **delivered** |

Not yet tried: stok sweep at TP8 (k=8 rejected on Flash, but acceptance here runs 6.9/7 on saturated categories), SEQS=16 (rejected on Flash), MAXLEN 131K (KV pool supports it; 32K chosen for battery parity).

## Appendix: failure ladder (all fixed; kept for reuse)

- **A1 — PP × DFlash:** any `--pipeline-parallel-size > 1` + speculative config → `NotImplementedError: Pipeline parallelism is not supported for this model` (drafter lacks `SupportsPP`). Structural; pick PP1 or skip speculation.
- **A2 — NCCL fd exhaustion at TP8:** 8-rank full mesh (7 peers × channels × RoCE QPs) exceeds default container fd limits → `NCCL WARN … Too many open files` → coordinated `WorkerProc failed to start` on all ranks. PP topologies stay under the limit (point-to-point neighbors only). Fix: `--ulimit nofile=1048576:1048576`.
- **A3 — CUTLASS block-FP8 vs TP8 shard shapes:** a fused hybrid projection shards to N=3392 at TP8; 3392 % 128 = 64 violates the CUTLASS tile requirement. `can_implement` passes at init, first graph execution dies: `cutlass_gemm_caller.cuh:52, Invalid status` (surfaces as `Engine core initialization failed`). At TP2 the shard is 6784 (÷128 = 53) — which is why the 2-node Flash recipe never hits it. Fix: `--linear-backend triton` (this vLLM build filters candidates in `model_executor/kernels/linear/__init__.py`; falls back per-layer with a warning when a layer type lacks the requested backend). MoE stays on marlin.
- **A4 — first-boot Triton JIT:** ~43 min compile before the API opens (vs ~12 min cutlass boot). Persist `$CACHE/triton`; size API-wait loops ≥60 min for cold caches.
- **A5 — GMU admission at 131K:** KV leftover must hold at least one max-length request. GMU 0.85/131K left 5.0 GiB/rank → `_check_enough_kv_cache_memory` ValueError. Either cut `--max-model-len` or raise GMU; at TP8 both knobs are comfortable (0.80/32K → 19.24 GiB/rank).
- **A6 — orphaned GPU memory on GB10:** a SIGKILLed NCCL-crashed container leaked ~21 GiB on the head, invisible to `ps`/`free`/`nvidia-smi --query-compute-apps` (reads 100.79/121.69 GiB free forever after). Only a node reboot reclaims it; until then, size GMU against the *measured* free, not the nominal (0.88 → 0.80 here).
- **A7 — ops footgun:** `ssh host 'pkill -f "vllm serve"'` self-matches the remote wrapper shell (and the local one when the terminal host is itself a target). Use `pkill -f "vllm [s]erve"`, and check `hostname` before fleet-wide loops — terminal sessions can drift between nodes after context compaction.

## Node notes

- **1d49** is ComfyUI-dedicated; used here as rank 7 under explicit owner authorization with ComfyUI paused. Restore ComfyUI after teardown.
- **04af** carries the A6 leak until its next reboot; GMU 0.80 accommodates it.
