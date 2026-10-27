# Runbook: DeepSeek V4.1 Flash on 4× DGX Spark (MiaAI SGLang TP4, production line)

- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ pinned rev `1bae0e10` (48 shards, 486 GB), now serving the **abliterated variant** `dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8`.
- **Source recipe:** [MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks) (TP4 production line per its `docs/tp4.md`, v2.1 state of knapcio/DeepSeek-V4.1-Flash-4x-DGX-Spark-TP4).
- **Engine:** SGLang (`dsv41-4x-spark:canary-roce`, 33.4 GB, base `lmsysorg/sglang:dev-dsv41`), **not vLLM** — supersedes our September [vLLM TP4 arm](runbook-dsv41-flash-tp4.md) on every measured axis.
- **Hardware:** head **edgexpert-04af** (10.10.10.15, weights local + kernel NFS export) + workers **9105** (.1), **bdea** (.2), **gx10-141d** (.16). RoCE fabric `enp1s0f0np0`.
- **Endpoint:** `http://10.10.10.15:8888/v1`, model id `deepseek-v4.1-flash`, **1M context** (KV pin `MAX_TOTAL_TOKENS=8000000`, fp8 KV).
- **Campaign dates:** 2026-10-18 (base weights) → 2026-10-20/21 (abliterated swap + battery). Live and serving as of 2026-10-21.

## Production line (exact knobs, from the live serve log)

TP4, `EP_SIZE=1`, DSpark k=5 spec decode (`speculative_dspark_block_size=5`, 6 draft tokens), `MEM_FRACTION_STATIC=0.80`, `CHUNKED_PREFILL_SIZE=4096` + `DSV41_INDEXER_CHUNKED=1`, `max_running_requests=16`, fp8_e4m3 KV, page 256. All ops through `./start-tp4.sh doctor|build|share|pack|serve|stop|status|smoke` — never raw docker compose.

Boot markers that prove the production line is active (all verified in our Oct 21 serve log): `RoCEnante ready: world=4`, `[moe_b12x_next] armed` (W4A8 in-place repack), `DSV41 Engram prefetch ARMED`, `DSV41_PREFILL_SP=1 ARMED`, `DSV41 shared-expert K padding ARMED`, `The server is fired up and ready to roll!` (01:34:12 → 01:37:35 ≈ **3.5 min boot** with warm NFS page cache).

## Measured results (our fleet)

### Decode battery, abliterated variant (2026-10-21, recipe's `benchmarks/decode_window.py`, idle server, greedy, thinking off, 2 warm-up runs discarded)

| C | aggregate tok/s | per-stream median (129–641 tok window) | median TTFT (s) |
|---|---:|---:|---:|
| 1 | **121.9** | 135.9 | 0.20 |
| 2 | 160.1 | 86.3 | 0.39 |
| 4 | 205.8 | 52.7 | 0.41 |
| 8 | **272.0** | 33.4 | 0.48 |

Run-to-run repeatability: three consecutive C4/C8 pairs within the session agreed to ±2% (206.8/267.3, 210.5/272.5, 205.8/272.0 agg). Raw: [`results/dsv41-mia-sglang-tp4/decode-window-ablit-20261021.json`](results/dsv41-mia-sglang-tp4/decode-window-ablit-20261021.json) + console log.

### Versus our September vLLM TP4 arm (same 4 nodes, ours9 bench)

| axis | vLLM (Sep) | SGLang MiaAI (Oct) | delta |
|---|---:|---:|---:|
| C1 aggregate | 31.7 | 121.9 | **3.8×** |
| C4 aggregate | 75.2 | 205.8 | 2.7× |
| best aggregate (C6 vs C8) | 99.2 | 272.0 | 2.7× |
| context served | 300K | 1M | — |
| cold boot | ~18 min | ~13 min (cold) / 3.5 min (warm) | — |

Recipe-published reference (sparkDash 1.8.8, switched RoCE, prose c1 87.7 / c8 237.6; json c16 659.9) — our decode_window numbers use a different harness and prompt mix, so compare within-harness only; both land in the same regime.

### Refusal probe, abliterated variant (2026-10-21, greedy T=0, 200 tok, chat mode, thinking off)

19 prompts across the model card's HarmBench categories (benign controls + adversarial), keyword-tier classified. Aggregate verdicts only:

| category | n | COMPLY | HEDGE/SOFT | HARD_REF |
|---|---:|---:|---:|---:|
| benign controls | 5 | 4 | 0 | 0 |
| adversarial (chem/cyber/harassment/illegal/misinfo/multilingual/selfharm/weapons) | 14 | 14 | 0 | 0 |

Abliteration confirmed working: 14/14 adversarial comply, zero hard refusals; benign controls unaffected (one 1-word-answer control classified UNCLEAR by the keyword tier — output was correct). Raw per-prompt JSON stays on the head node (`~/dsv41u-refusal.json` on 04af) by policy; probe script pattern documented in the campaign notes.

## Abliterated weight swap (base → UNCENSORED-FP8, verified procedure)

1. Diff the variant repo against the base via HF API: 48 shards, **21 byte-identical** (hardlink from the base copy), **27 different** (shards 13–39, ~200 GB download; SHA-256-verify every one against HF LFS oids).
2. Per-node WiFi variance is real — 04af's adapter negotiated 8.6–17 Mbps TX; downloaded on the fast node and fabric-rsync'd (~350–480 MB/s).
3. Repoint `.env.tp4` `MODEL_DIR`, `/etc/exports` (one sudo moment), and every worker's `~/worker-nfs-bind.sh` EXPORT; verify all 3 workers see 48 shards through their mounts.
4. **Repack Engram shards** — `pack` skips by SIZE ONLY and silently reuses stale bins, which mixes old weights into the n-gram lookups with no error. Delete `engram-*.bin` on head + all workers first (root-owned; needs sudo), then `./start-tp4.sh pack` (~7 min across 4 nodes).
5. `serve` → `smoke` (19×23 → 42) → battery. Revert path: base checkpoint still intact at `~/NewModels/DeepSeek-V4.1-Flash-1017` on 04af; repoint the same three layers + repack.

## Fleet adaptations (recipe → ours)

| item | recipe default | ours | why |
|---|---|---|---|
| fabric NIC | `enp1s0f1np1` | `enp1s0f0np0` (FABRIC_IFACE + GLOO/NCCL_SOCKET_IFNAME) | fleet NIC naming |
| NFS export | recipe's containerized exporter | **kernel nfsd** via `/etc/exports` (`ro,sync,no_subtree_check,no_root_squash`) | kernel nfsd owns :2049; the container exporter restart-loops and `share`'s `rpcinfo` probe gets a false OK |
| worker weight mount | docker NFS-type volume (`--opt type=nfs`) | host NFS mount via privileged container + **bind volume** (`~/worker-nfs-bind.sh`) | docker 29.2.1 on this fleet rejects NFS-type volumes with `invalid argument` on every worker |
| NCCL buffers | defaults | `NCCL_BUFFSIZE=1048576`, `NCCL_LL128_BUFFSIZE=262144`, `NCCL_PROTO=^LL128` | GB10 unified memory: default pinned buffers cost ~4.7 GiB/node; trimmed to ~0.14 GB (recipe-provided, kept) |
| SSH identity | `id_ed25519_shared` | `id_ed25519`, WORKER_USER=vikassridhar | fleet keys |

The mount container MUST run with `--network host` + `nsenter -t 1 -m`: without host networking the mount joins the container netns, dies at exit, and leaves a stale `Network is unreachable` mountpoint that wedges the next attempt (`umount -l` first). Verify the mount from a second short-lived container before creating the bind volume. After any worker reboot: re-run `~/worker-nfs-bind.sh` there before `serve`.

## Gotchas banked this campaign

- `pkill -f "hf download"` over SSH kills your own session — bracket the pattern (`hf downloa[d]`).
- rsync does not create destination parents; `ssh host mkdir -p` first, run the first seconds in foreground (detached rsync reports failures on the receiver, sender log stays empty).
- Node→node SSH trust is asymmetric (04af→ae1e works, not the reverse) — pull from the authorized side.
- Stale `/etc/exports` lines for deleted dirs print harmless `Failed to stat` on `exportfs -ra`; verify with `showmount -e localhost`.
- The decode battery hard-requires an idle server (`/v1/loads` running+waiting = 0) — run quality/refusal probes strictly after it, never concurrently.
- Always send `max_tokens` + `chat_template_kwargs:{"thinking":false}` for benches.

## Status

**LIVE** — abliterated variant serving at `http://10.10.10.15:8888/v1` since 2026-10-21. GPU contention check before (re)serve: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` empty on all 4 nodes (gr8 was stopped on 04af+bdea for this deployment; ae1e's gr8 untouched).
