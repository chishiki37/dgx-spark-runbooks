# Runbook — GLM-5.3 EXL3 (ablit) on 4× DGX Spark: cuda-exl3 + DFlash2 stack

Update of `runbook-glm53-exl3-abliterated.md` (MTP stack). Serves `drowzeys/keys-GLM-5.3-EXL3-Abliterated`
(GLM-5.3 753B, 3-bit EXL3, ablit baked) with the publisher's prebuilt image **`ghcr.io/drowzeys/glm53-exl3-tp4:latest`**
(cuda-exl3 MoE kernels + DFlash2 spec decode + EXL3 overlay baked in) + stock **`incoai/GLM-5.3-DFlash2`** drafter.
Upstream recipe: `drowzeys/keys-GLM-5.3-EXL3-3.0BPW-abliterated-vLLm-cuda-Exl3`.

Campaign repo: `chishiki37/glm-5.3-exl3-abliterated-4x-dgx-spark` (reports 01 + 04; 04 = this stack).
Measured 2026-09-07 on lane 04af/bdea/1d49/gx10 (rail-1, 10.10.10.x): **200K/DCP1: C1 16.44 / C8 agg 56.36**;
**1M/DCP4: C1 13.84 / C8 agg 41.91** (vs MTP-k1: +31% / +42% at 200K-equivalent).

## Prerequisites (one-time, persists across reboots)

1. Weights on head `~/models/keys-GLM-5.3-EXL3-Abliterated` (41 shards, 308 GB), NFS-exported ro over
   NFSoRDMA port 20049 to the fabric subnets; workers mount `/mnt/glm53exl3`; head uses a **local bind mount**
   of the same path (never self-mount the export).
2. Image on ALL 4 ranks: `docker pull ghcr.io/drowzeys/glm53-exl3-tp4:latest` on the head, then
   `docker save | ssh <worker> docker load` per worker (verify per rank — a silent partial ship boots 3 ranks fine
   and the 4th dies at overlay validation).
3. Drafter as a plain dir on ALL ranks at the SAME absolute path
   (`~/models/drafters/GLM-5.3-DFlash2`: config.json + model.safetensors, 4.9 GB, sha256-verified).
   On the head, `hf download` may deadlock on IPv6 (SYN-SENT, dead v6 route) — use `curl -4` per file.
4. Sudoers NOPASSWD for the flusher on every rank (`/etc/sudoers.d/vllm-flusher`: `tee /proc/sys/vm/drop_caches`,
   `sysctl -w vm.vfs_cache_pressure=200`); `vfs_cache_pressure=200` persisted via `/etc/sysctl.d`.
5. Launcher `~/glm53-mia/start-tp4.sh` patched (head `-e` list + spec-JSON template). **Verify before every boot**
   (see Checklist) — an unpatched launcher silently drops `DFLASH_REJECT`/`KEYS_CUDA_EXL3`.
6. Per-rank RoCE GIDs re-probed live before EVERY boot (drifts across reboots): 04af=3, bdea=5, 1d49=3, gx10=3
   at last probe. Probe loop is in the campaign repo scripts / RECIPE.md upstream.

## Champion .env.tp4 (200K / DCP=1 / k5 / block)

Rail-1 IFs `enp1s0f0np0`/`rocep1s0f0`, IPs 04af=.15 (head) / bdea=.2 / 1d49=.11 / gx10=.16.
Key deltas vs the MTP-era env:

```
IMAGE=ghcr.io/drowzeys/glm53-exl3-tp4:latest
SPEC_METHOD=dflash
DFLASH_MODEL=incoai/GLM-5.3-DFlash2
DFLASH_TOKENS=5
DFLASH_DRAFT_TP=1          # MUST be explicit — a leftover in .env (draft TP=2) hard-crashes vLLM
DFLASH_REJECT=block        # requires the launcher patch; assert in engine log
KEYS_CUDA_EXL3=1           # requires the launcher patch; assert in engine log
LOCAL_DFLASH_DIR=/home/vikassridhar/models/drafters/GLM-5.3-DFlash2   # same path on all ranks
GPU_MEM_UTIL=0.75          # engine budget 91.2 GiB; 73.2 weights + 4.9 draft + 13.1 KV ≈ 0.3 slack
MAX_MODEL_LEN=200000
MAX_NUM_BATCHED_TOKENS=4096
MTP_TOKENS=0
EXTRA_ARGS="--kv-cache-memory-bytes 14092861440 --cudagraph-capture-sizes 6 12 18 24 --async-scheduling --cudagraph-metrics --compilation-config.pass_config.fuse_allreduce_rms=true --no-enable-prefix-caching"
```

Layout arms (edit two lines only):
- 500K/DCP2: `MAX_MODEL_LEN=500000`; EXTRA_ARGS prepend `--decode-context-parallel-size 2`, KV bytes → 16900000000
- 1M/DCP4: `MAX_MODEL_LEN=1000000`; EXTRA_ARGS prepend `--decode-context-parallel-size 4`, KV bytes → 16900000000

## Boot

1. Teardown old containers on all ranks (`start-tp4.sh stop` + `docker rm -f` the 4 names on each host).
2. Start threshold flushers on all ranks: `setsid bash ~/flusher.sh </dev/null >/dev/null 2>&1 &`
   (drops caches only when MemAvailable < 8 GiB, poll 30 s; stop sentinel /tmp/flusher_stop).
   **Never** use a fixed-period flusher during NFS load — livelock (12 s cadence = 3.5 h zero progress).
3. One `sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches` per rank.
4. Launch: `SKIP_PULL=1 SKIP_SHIP=1 SKIP_BUILD=1 SKIP_DOWNLOAD=1 SKIP_SYNC=1 SKIP_OVERLAY_VERIFY=1 ./start-tp4.sh`
   (assets are pre-staged; drafter comes from LOCAL_DFLASH_DIR, not the HF cache).
5. Health gate: `curl :8888/v1/models` until GLM-5.3-EXL3 appears (cold ≈ 11–15 min, warm ≈ 8–11).
6. **Engine-log checklist before trusting any number** (grep the launch log):
   - `"rejection_sample_method": "block"` in speculative_config + `block_verify=True` warm-up line
   - `num_speculative_tokens: 5`, `draft_tensor_parallel_size: 1`
   - `cudagraph_mode: FULL_AND_PIECEWISE` (any PIECEWISE-only downgrade = a dynamic spec schedule leaked in)
   - `[keys32] cuda-exl3 MoE kernel engaged: 75 layers (expected 75)`
   - `GPU KV cache size: N tokens` ≥ max_model_len
   - `enable_prefix_caching: False`
7. Warm-up: the FIRST request after boot pays the DCP4 graph warm-up (29–41 s, first C1 rep can collapse
   to ~7 tok/s). Fire one ~256-token completion and discard before benchmarking or serving.
8. Stop flushers after health: `touch /tmp/flusher_stop` on all ranks.

## Battery

`python3 ~/glm53exl3/glm53exl3_bench.py <out.json>` — C1×3 median + C4×2 + C8×2, 267-tok prompt,
256 out, temp 0, thinking OFF. Bank results to `~/` (home), never `/tmp` (tmpfs, wiped on reboot).

## Troubleshooting quick list

- `speculative_draft_tensor_parallel_size=2 cannot be other value than 1 or target` → stale
  `DFLASH_DRAFT_TP` in `~/glm53-mia/.env` (`.env.tp4` must override explicitly with =1).
- Spec config shows `"standard"` / no `keys32` engagement line → launcher unpatched or old image on one rank.
- `no snapshots under .../models--incoai--GLM-5.3-DFlash2` → `LOCAL_DFLASH_DIR` missing from `.env.tp4`
  (launcher fell back to the HF-cache path).
- One rank wedged during NFS load, others fine → check its flusher is actually running (`pgrep -f flusher.sh`
  per rank — ssh-heredoc-launched flushers die silently and non-deterministically).
- Head OOM during load → head is reading over NFS instead of the local bind mount (wedge doctrine).
