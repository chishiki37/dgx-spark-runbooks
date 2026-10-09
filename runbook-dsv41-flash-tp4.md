# Runbook: DeepSeek V4.1 Flash on 4× DGX Spark (vLLM TP4, Engram-on-disk, DSpark k=5 spec decode)

- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash` — 552B backbone MoE / 769B with Engram n-gram tables, MXFP4 experts + MXFP8 dense, 510.3 GB, 48 shards, 384 routed experts top-6 + 1 shared, 1M native context, multimodal (`DeepseekV41ForCausalLM`).
- **Source recipe:** [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) (boot9 era), reproduced on our fleet 2026-09-10/11.
- **Hardware:** 4× NVIDIA DGX Spark (GB10) — head **edgexpert-04af** (10.10.10.15, weights local, NFS export) + workers **9105** (.1), **bdea** (.2), **gx10-141d** (.16) via docker-managed NFS volumes. RoCE fabric, fleet-proven NCCL env.
- **Image:** `vllm-dsv41:ours-a5` (23.3 GB) on all 4 ranks — vLLM `dsv41-feat` branch on merge-base `nightly-8a728663`, FlashInfer 0.7.0rc1, Engram-on-disk patches, arm64 `_C_stable_libtorch`.
- **Endpoint:** `http://10.10.10.15:8000/v1`, model id `deepseek-v4.1-flash`, 300K context.
- **Campaign date:** 2026-09-10 → 09-11 · results banked on 04af `~/ds41-vllm/results/ours9/`, launcher `~/boot_ours.sh` + `~/dsv41-tp4-ours.sh`, campaign dir `~/ds41-campaign/`.

## Why 4 Sparks (topology constraint, verified from config + converter asserts)

MP must divide BOTH `n_experts=384` AND `mtp_n_experts=128` → valid MP ∈ {1, 2, 4, 8, 16}. **TP5/TP6 are mathematically impossible.** TP4 = ~128 GB/node as-shipped doesn't fit a 121 GiB GB10 — it fits **only** with the Engram-on-disk patch (the two ~101 GB n-gram tables stay in the safetensors files, read via preadv, rows staged in `prepare_inputs` so the forward stays CUDA-graphable; stock vLLM keeps them in host RAM, which IS the GPU pool on GB10). TP8 = ~64 GB/node fits comfortably (that was our Path-A reference-impl probe).

## Measured results (ours9 bench, 2026-09-10; prompt set v1, temp 0, thinking off)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| 1 | 31.7 | **36.2** | 0.56 |
| 2 | 54.3 | 31.8 | 0.77 |
| 3 | 64.7 | 25.0 | 0.57 |
| 4 | 75.2 | 21.6 | 0.67 |
| 5 | 88.0 | 20.4 | 0.68 |
| 6 | **99.2** | 19.2 | 0.78 |

- Upstream reference (tonyd2wild boot9, 4× Spark): C1 33.9 agg / 39.2 per-stream, C6 98.0 — **we reproduced parity at concurrency, ~93% at C1**. The residual C1 gap is structural, not a config bug: his fleet is uniform ConnectX-7 200G; ours must include head 04af, a 100G-cage node (200G cage = 9105/bdea/141d only, and the head can't move — it holds the weights + NFS export).
- Cold prefill: 468 / 369 / 538 / 615 tok/s at 2,950 / 11,592 / 46,810 / 93,335 prompt tokens (non-monotonic — the 8K-target row dips; raw rows in `bench-ours9.json`).
- Per-category C1 spread 20.2 (narrative) → 54.6 (format) — spec-decode acceptance is content-driven.
- Live DSpark acceptance (k=5): 3.57 tok/step; per-position 77.9 / 60.3 / 46.4 / 39.1 / 33.4% (pos4 marginal: +0.33 tok/step per extra drafter pass — spec depth is the primary decode lever and was never swept upstream; k=5 is the shipped default).
- Correctness: vision **correct** in vLLM lane ("carrots with green stems" probe), tools on, 300K ctx. Raw per-category tables: `results/dsv41-flash-tp4/bench-ours9.{md,json}`.

## Serve config (exact boot9/incumbent knobs — restore recipe)

```bash
IMAGE=vllm-dsv41:ours-a5 PATCH_DIR=~/patches/dsv41-boot3 GMU=0.80 MAXLEN=300000 SEQS=8 \
MAX_BATCHED=8192 EAGER=0 CUDAGRAPH_MODE=FULL_AND_PIECEWISE SPEC=dspark SPEC_K=5 ENGRAM_DISK=1 \
TEXT_ONLY=0 THINKING=false PARSERS=1 RUST_FE=0 \
NCCL_EXTRA="-e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton" \
VLLM_EXTRA='--block-size 128 --limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1' \
bash ~/boot_ours.sh
```

Non-negotiables: `--block-size 128`; FULL_AND_PIECEWISE CUDA graphs with **exact** capture sizes (auto-derived from K: multiples of K to K×SEQS ∪ multiples of K+1 to (K+1)×SEQS — padded spec batches hang SM120 sparse MLA, FlashInfer #5015; never hand-set CG_SIZES when sweeping K).

## Fleet adaptations (his recipe → ours)

| item | upstream | ours | why |
|---|---|---|---|
| NCCL env | `NCCL_IB_GID_INDEX=3` + ADDR_RANGE pins (his CX-7 fabric) | no GID pin, `NCCL_CROSS_NIC=1`, `NCCL_IB_HCA=rocep1s0f0`, `NCCL_SOCKET_IFNAME=enp1s0f0np0` | our NICs carry extra GIDs (2nd port 192.168.100.x, rail-2 10.10.20.x) — GID 3 unroutable on bdea/gx10 → `ncclCommInitRank: unhandled system error` |
| worker weights | local per node | docker-managed NFS volumes from head (`addr=`, `nfsvers=3`) | zero worker sudo |
| launch wrappers | `launch/bootN-go.sh` | `~/boot_ours.sh` (worker-first fan-out) + `~/dsv41-tp4-ours.sh` | his wrappers are STALE on our fleet: boot9-go.sh exports `IMAGE=vllm-dsv41:overlay5` which exists only on the build head — workers carry `ours-a5`; one rank's missing image kills the whole fan-out (set -e) |
| sudo drop_caches | required | best-effort (staged `/tmp/.spw` when present) | no passwordless sudo on our nodes |
| tokenizer mode | `--tokenizer-mode deepseek_v41` via tag-name rule | not needed | that rule applies only to the raw official-image shortcut; the overlay tree registers the mode itself (live-proven) |

## Deploy steps (condensed; full detail in campaign records)

1. **Weights** (510.3 GB) on the head's local disk (`~/models/DeepSeek-V4.1-Flash` on 04af); head exports it read-only over NFS to both rails (one sudo moment: `/etc/exports` + `exportfs -ra`). Workers create the docker NFS volume (`--opt o=ro,addr=10.10.10.15,nfsvers=3,tcp,rsize=1048576,...` — `addr=` and `nfsvers=3` spellings are mandatory or the mount fails with bare `invalid argument`).
2. **Image chain** (~11 min total on-fleet): base `vllm/vllm-openai:deepseekv41-flash-0909-arm64` → overlay3 (FlashInfer 0.7.0rc1 pinned submodules — 0.6.18 lacks SM120 sparse-MLA decode for topk=1152) → overlay4 (mxfp8 prebuilt, MAX_JOBS guard) → overlay5 (sparse_mla runtime-env rebuild + GPU verify). `Dockerfile.overlay` must be copied as `Dockerfile` in the build context; `build_stable_ext.sh` runs on the HOST (wraps its own `docker exec`); the built `_C_stable_libtorch.abi3.so` must be copied into the cloned tree's `vllm/` so the overlay's `COPY vllm/` carries it. **Patch sets are tree-pinned** — the official image diverged from the `dsv41-feat` base (`model.py` imports `gather_engram_hashes`, absent from the patched `engram.py` → registry inspect ImportError); verify with grep before building. Ship to workers via `docker save | ssh docker load` (save runs on the SOURCE).
3. **Boot** ≈ 18 min, worker-first fan-out, stop-first head-first. Launcher gates: MemAvailable ≥ 100 GiB, patch mounts.txt completeness, model presence. While serving, MemAvailable = 6–8 GiB on ALL ranks (unified memory) — any sweep chain must teardown → wait mem-free → boot.
4. **Smoke** (`~/ds41-campaign/smoke9.py`), then bench from a sibling node over the fabric (prompt set v1, temp 0).

## Path A (reference-impl TP8) — correctness probe, superseded

DeepSeek's HF repo ships a reference PyTorch stack (tilelang 0.1.8, torch ≥ 2.10, torchrun across 8 Sparks, 67.2 GB/rank). Text correctness **perfect** (EN+ZH, temp 0.6); **vision broken** (BOS-led token garbage — suspect ViT TP sharding; treat reference-impl vision output as untrustworthy). Useful findings: stock `convert.py` OOMs (accumulates all ranks in RAM; streaming variant peaks ~64 GB), container-written shards are root:root 0600 (chmod via privileged container before rsync), NCCL needs `--ulimit memlock=-1 --cap-add IPC_LOCK` (found by diffing HostConfig against a known-good container — the fastest debug move for container-env NCCL failures).

## Known gaps / open follow-ups

- Quality suite (GSM8K/MBPP/IFEval/MMLU…) **never run** on this lane — speed battery only.
- 300K needle test not run; spec-depth (K) sweep never done (k=5 shipped default).
- C1–C3 ~7% gap vs upstream is attributed to the 100G-cage head leg; all-200G TP4 placement is impossible on this fleet (cage map above).

## Appendix: bugs & fixes

1. **NCCL GID index poisoning (fleet-local):** his `NCCL_IB_GID_INDEX=3` dies on exactly the nodes whose GID tables differ (`show_gids` to inspect). Diagnose by finding the FIRST-dying rank, not cascade victims ("remote process exited" lines are noise).
2. **Stale upstream launch wrappers:** boot9-go.sh → missing `$HOME/boot_dsv41.sh` + head-only image tag → `pull access denied` on workers, and with set -e fan-out, production stays down until relaunch. Keep OUR wrappers as ground truth.
3. **Padded CUDA-graph capture sizes hang SM120 sparse MLA** (FlashInfer #5015) — exact sizes only (auto-derived; see serve config).
4. **GB10 EC clock-latch:** stuck <1 GHz clocks (only a 30–60 s power-unplug clears; `nvidia-smi` looks clean). Burn-test before any benchmark: 8192³ fp16 matmuls ×15 s → healthy = 93–95 TFLOPS. All 8 nodes measured healthy for this campaign.
5. **`VERIFY mxfp8: MISS`** after overlay5 is a cache-path quirk, benign — launcher pins `MAX_JOBS=2`/`FLASHINFER_NVCC_THREADS=1` so worst case is bounded one-time JIT at first request.
