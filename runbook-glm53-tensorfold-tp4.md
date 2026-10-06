# Runbook: GLM-5.3 (753B full) on 4× DGX Spark via TensorFold TP4 (drowzeys/keys recipe)

- **Model:** `drowzeys/keys-GLM-5.3-EXL3-2.75BPW` (258 GiB, 89 files) — keys35 quant: routed experts EXL3 mixed-K mul1 (2–4 bit, per-expert table), non-experts EXL3 5-bit, MTP 8-bit, quantized from `zai-org/GLM-5.3-BF16`. Model type `glm_moe_dsa`.
- **Engine:** [ashhart/TensorFold](https://github.com/ashhart/TensorFold) via image `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-04-opt` (branch `glm53-tp4-opt` built from source).
- **Source recipe:** [drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark) — one-shot flow `check → up → wait → bench`.
- **Campaign date:** 2026-10-06 · nodes **04af (rank0/API .15) + bdea (.2) + ae1e (.13) + 1d49 (.11)**, single 100G RoCE rail (`RAILS=1`). Repo `edgexpert-04af:~/tf-glm53-tp4`, checkpoint NFS-exported from 04af at `/mnt/glm53-exl3`, results `~/tf-campaign/glm53-tp4-tf/`.
- **Endpoint:** `http://10.10.10.15:8890/v1` (served name `glm-5.3-tf`)
- **Serving config:** `CONTEXT=140000 PARALLEL=1 RAILS=1 IF=enp1s0f0np0`, DFlash2 drafter (4096-slot ring, works at any context), thinking on.

## Measured results (2026-10-06)

### Recipe built-in bench (greedy, thinking on, request-time)

| metric | ours | recipe claim |
|---|---|---|
| prose decode | **44.2 tok/s** | ~41.8 |
| code decode | **38.8 tok/s** | ~38.1 |
| cold 32K prompt (25,470 tok) | **26.0 s (~980 tok/s prefill), needle PASS** | — |

### Our standard battery (prompt set v1, temp 0, thinking off; C1 only — engine serves one stream at a time at PARALLEL=1)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| 1 | **37.1** | 40.7 | 0.411 |

Per-stream C1: coding 43.9 · format 43.8 · reasoning 43.3 · json 41.9 · math 40.5 · summary 38.4 · prose 38.0 · narrative 35.6 · ceiling-count 51.7 (excluded).

Cold prefill: **1,107 @2K / 1,099 @8K / 1,052 @32K tok/s** (3.8K/15.2K/60.9K actual tokens).

### vs our previous GLM-5.3 full-size arms (same fleet; prior vLLM arms ran on 04af/bdea/1d49/gx10)

| metric | vLLM EXL3-ablit TP4+DCP4 (09-05) | vLLM cuda-exl3+DFlash2 (09-07) | **TensorFold TP4 (this)** |
|---|---|---|---|
| C1 | 12.6 | 16.4 | **37.1** |
| C8 agg | 39.7 | 56.4 @200K | n/a (single-stream engine) |
| prefill @32K | — | — | ~1,050 |
| context | 1M | 200K–1M | 140K (DCP off) / 1M (DCP4, ~−10% decode) |

**Headline: TensorFold is 2.3–2.9× our best vLLM arm for single-stream GLM-5.3 decode** (37.1 vs 12.6–16.4 C1) — and it beat the recipe's own published numbers. Trade-off: no concurrent streams at this context (KV per rank is large), vs vLLM's C8 batching. Best-in-class single-user experience for the 753B flagship on 4 Sparks; keep the vLLM int4-int8mix TP4 arm (26.3 C1 / 89.2 C8 agg, 200K) for multi-user.

## Deploy steps (deltas from the recipe README)

1. Pull image on all 4 nodes; mount checkpoint at the **same path** everywhere (NFS export from rank0; rank0 itself can't NFS-mount its own export → **bind-mount** `/home/vikassridhar/models/keys-GLM-5.3-EXL3-2.75BPW → /mnt/glm53-exl3` via privileged container + nsenter).
2. Worker NFS mounts must use the **host net namespace** (`nsenter -t 1 -m -n`) or mount fails with "Network is unreachable".
3. rank0 must be able to SSH to its own fabric IP (one-shot fans out to all NODES incl. self) — append its own pubkey to authorized_keys if not.
4. `check → up → wait → bench` with env: `NODES='10.10.10.15 10.10.10.2 10.10.10.13 10.10.10.11' MODEL=/mnt/glm53-exl3 RAILS=1 IF=enp1s0f0np0`.
5. Sysctls per recipe: `vm.compaction_proactiveness=0`, `vm.swappiness=1` on every node.

## Appendix: bugs & fixes (failure ladder)

1. **Wrong checkpoint rejected** — our existing `keys-GLM-5.3-EXL3-Abliterated` (308G) fails the engine's reader: it's a mixed-K variant with 16-bit heads TensorFold's CUDA `glm_moe_dsa` loader doesn't support. Fix: use the recipe's tested `keys-GLM-5.3-EXL3-2.75BPW` (verify `quantization_config` in config.json: `exl3 / mul1 / keys35-v1`).
2. **NCCL error 5 "no socket interface found"** — recipe defaults `IF=enp1s0f1np1`; on this fleet the live fabric NIC is **`enp1s0f0np0`** (second port is DOWN). Pass `IF=` explicitly; check with `ip -br addr | awk '/10\.10\.10/'` on every rank.
3. **KV cache OOM at graph capture** (`context 140003 × 1 needs 11.7 GiB/rank, 1.3 GiB free`) — GB10 unified memory: the ~10-min NFS weight load re-fills page cache on every rank, and the allocator counts only *free* memory. Dropping caches once **before** boot is not enough. Fix: run a **continuous drop_caches loop (every 25 s, all ranks, via privileged container nsenter — no sudo needed) during the entire load phase**; boot then passed with the default 140K context.
4. `one-shot.sh down` between attempts — leftover rank containers hold GPU memory and the next `up` silently reuses/fails them.
