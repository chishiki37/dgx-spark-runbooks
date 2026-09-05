# Runbook — keys-GLM-5.3-EXL3-Abliterated on 4× DGX Spark (TP4 + DCP4, 1M ctx)

| | |
|---|---|
| **Model** | `drowzeys/keys-GLM-5.3-EXL3-Abliterated` (GLM-5.3 753B, 3-bit EXL3, abliterated; 330.18 GB / 79 files / 308 GiB on disk) |
| **Hardware** | 4× NVIDIA DGX Spark (GB10, 121 GiB unified, sm_121a) — head `10.10.10.15` (edgexpert-04af) + workers `10.10.10.2`, `10.10.10.11`, `10.10.10.16` |
| **Performance (measured 2026-09-05, winner k1)** | C1 **12.57 tok/s** · C4 **32.68 tok/s agg** · C8 **39.74 tok/s agg** · TTFT ~2.3 s @267-token prompt |
| **Context** | 1,000,000 tokens (KV pool 1,029,690 tokens, fp8, 13.1 GiB pinned) |
| **Endpoint** | OpenAI-compatible `http://<head>:8888/v1` (model id `GLM-5.3-EXL3`) |
| **Deployment dir** | head: `~/glm53-mia` (launcher) + `~/glm53exl3` (campaign) · weights `~/models/keys-GLM-5.3-EXL3-Abliterated` |
| **Image** | `glm53-exl3-tp4:keys3` (chain: MiaAI-Lab EXL3-vLLM base → dcp → mul1 → v145 → keys → keys3; exllamav3 1.4.5 built for sm_121a) |

## Prerequisites

1. 4× DGX Spark on one RoCE fabric, passwordless SSH head→workers (fabric IPs above), Docker on all ranks.
2. MiaAI-Lab EXL3-vLLM serving repo on head (`~/glm53-mia`, provides `start-tp4.sh` launcher) + drowzeys patch-chain Dockerfiles (`~/glm53exl3/serving/Dockerfile.{dcp,mul1,v145,keys,keys3}`).
3. NFS kernel server on head; workers mount over RDMA.
4. HF access to the model repo. A curl-based downloader (`scripts/hf_curl_download.py` pattern) is more reliable than `hf hub download` on this fleet.
5. sudo password staged at `/tmp/.spw` on every rank (used non-interactively by flusher/mount helpers).

## Step-by-step

### 1. Download weights on the storage-rich node (head)

```bash
mkdir -p ~/models && cd ~/glm53exl3
python3 hf_curl_download.py drowzeys/keys-GLM-5.3-EXL3-Abliterated \
  ~/models/keys-GLM-5.3-EXL3-Abliterated --workers 4 2>&1 | tee download.log
python3 verify_sizes.py ~/models/keys-GLM-5.3-EXL3-Abliterated   # manifest check, 79 files
```

### 2. Export over NFS; mount on workers via RDMA

Head `/etc/exports` (append, then `sudo exportfs -ra`):

```
/home/vikassridhar/models/keys-GLM-5.3-EXL3-Abliterated 10.10.10.0/24(ro,sync,no_subtree_check,no_root_squash)
```

Each worker:

```bash
sudo mkdir -p /mnt/glm53exl3
sudo mount -t nfs -o ro,vers=3,proto=rdma,port=20049,rsize=1048576,wsize=1048576,hard,timeo=600 \
  10.10.10.15:/home/vikassridhar/models/keys-GLM-5.3-EXL3-Abliterated /mnt/glm53exl3
```

**Head does NOT mount its own export** (loopback wedge). Instead bind-mount:

```bash
sudo mkdir -p /mnt/glm53exl3
sudo mount --bind ~/models/keys-GLM-5.3-EXL3-Abliterated /mnt/glm53exl3
```

### 3. Build the image chain on head (aarch64 fixes required)

```bash
cd ~/glm53exl3/serving && export DOCKER_BUILDKIT=1
for stage in "Dockerfile.dcp dcp" "Dockerfile.mul1 dcp-mul1" "Dockerfile.v145 dcp-mul1-v145" "Dockerfile.keys keys" "Dockerfile.keys3 keys3"; do
  set -- $stage; docker build -f $1 -t glm53-exl3-tp4:$2 . || exit 1
done
```

exllamav3 1.4.5 needs three aarch64 patches to compile (x86 intrinsics in `avx*_target.cpp`, `EXL3_CPU_PAUSE` macro placement, hand-written `all_reduce_cpu_avx2.cpp` stubs replacing broken regex-generated ones) — details in `reports/03-bringup-ops-findings.md`.

### 4. Ship image to all workers — and VERIFY

```bash
for ip in 10.10.10.2 10.10.10.11 10.10.10.16; do
  docker save glm53-exl3-tp4:keys3 | ssh $ip 'docker load'
done
# verify on EVERY rank — a killed ship silently leaves the old image behind:
for ip in 10.10.10.2 10.10.10.11 10.10.10.16; do
  ssh $ip 'docker images | grep keys3'
done
```

### 5. Configure the launcher

`~/glm53-mia/.env.tp4` — key entries (full sanitized copy: `scripts/env.tp4`):

- `MODEL=drowzeys/keys-GLM-5.3-EXL3-Abliterated`, `IMAGE=glm53-exl3-tp4:keys3`
- `MAX_MODEL_LEN=1000000`, `GPU_MEM_UTIL=0.82`, `MAX_NUM_BATCHED_TOKENS=4096`
- Winner spec config: MTP speculative decoding with **1 draft token** (`SPEC_METHOD=mtp`), `EXL3_FUSED_MOE=1`
- RoCE GIDs are **live-probed per rank** by the launcher (bdea probed GID 5, others 3 — never hardcode)
- NIC: `enp1s0f0np0` on all ranks

### 6. Boot with cache-flusher sidecar (mandatory on cold cache)

```bash
# sysctl on all ranks:
sudo sysctl -w vm.vfs_cache_pressure=200 vm.swappiness=10
# flusher sidecar (boot window only — see scripts/boot_with_flusher.sh):
( while [ ! -f ~/glm53exl3/results/.stop_flusher ]; do
    for ip in 10.10.10.15 10.10.10.2 10.10.10.11 10.10.10.16; do
      ssh $ip 'cat /tmp/.spw | sudo -S bash -c "sync; echo 3 > /proc/sys/vm/drop_caches"' 2>/dev/null
    done; sleep 10; done ) &
# launcher needs nofile 1048576 (start-tp4.sh sets it; verify with ulimit -n)
cd ~/glm53-mia && ./start-tp4.sh
```

Health gate: `curl http://127.0.0.1:8888/v1/models | grep GLM-5.3-EXL3`. Cold boot ≈ 31 min; warm (page cache) ≈ 9–20 min. Stop the flusher (`touch results/.stop_flusher`) once healthy.

### 7. Smoke + benchmark

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-EXL3","messages":[{"role":"user","content":"Explain the role of the indexer in sparse MLA attention."}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
python3 glm53exl3_bench.py results/bench.json     # C1×3 + C4×2 + C8×2, 256 out tokens
```

## Key configuration details

- **Topology**: TP4 + DCP4, one rank per node; NCCL 2.30.7 over RoCE; `PYNCCL` all-reduce (no MNNVL multicast on GB10 — `SymmMemCommunicator: Device capability 12.1 not supported` warning is benign).
- **KV**: fp8, 1,029,690-token pool at 1M ctx (13.1 GiB pinned). Draft model max len auto-clamped 1048576 → 1000000.
- **Memory budget**: ~82.5 GB weights/rank + engine ≈ 100–103 GB/node at `GPU_MEM_UTIL=0.82`, leaving ~6–8 GiB host margin. Do not raise the util.
- **Abliteration is baked into the weights** — no runtime ablit flags.
- **Thinking mode**: `chat_template_kwargs: {"enable_thinking": false}` for benchmarks; both modes work.

## Performance reference (measured 2026-09-05, see reports/01)

| Config | C1 tok/s | C4 agg | C8 agg | TTFT C1 |
|---|---|---|---|---|
| **k1 — production pick** | **12.57** | **32.68** | **39.74** | 2.26 s |
| k0 (spec off) — throughput mode | 9.48 | 29.48 | 49.17 | 0.29 s |
| k3 (recipe default) | 12.12 | 18.79 | 29.74 | 2.51 s |

Prompt 267 tokens / 256 out tokens, temp=0, thinking OFF, warm engine.

## Troubleshooting pitfalls

1. **NCCL init "Error 3"** → `Too many open files`. Set nofile ulimit 1048576 before launch.
2. **OOM kill during weight load** → NFS page cache ate unified memory. Run the boot-window flusher + `vfs_cache_pressure=200`. Not needed post-boot.
3. **`exl3.py` validation error at engine init** → rank running the wrong (Mia-stock) image. Verify `docker images` on all ranks; re-ship.
4. **CUDA graph capture crash with unfused MoE** → `EXL3_FUSED_MOE=0` is broken on sm121a. Keep fused=1. Do not re-test.
5. **TCPStore broken-pipe cascade** → secondary symptom of a rank dying earlier; look at the FIRST failing rank's traceback, not rank3's store errors.
6. **Head NFS self-mount wedge** → head must bind-mount, never mount its own export.
7. **Automated secret scrubbers** mangle env files with TOKEN-like variable names → verify on-disk env after generation.
8. **First C1 run after warm-up is slow** (cold KV/graphs) → always take median of ≥3.
9. **Boot time doubles on cold cache** (520 s → 1860 s) → expected; the flusher trades boot speed for stability.

## Sources / credits

- Weights + abliteration + patch chain: **drowzeys** (`keys-GLM-5.3-EXL3-Abliterated`)
- EXL3-vLLM integration + TP4 launcher: **MiaAI-Lab**
- exllamav3 1.4.5: upstream project, aarch64/sm_121a build fixes by this campaign
- Campaign by Vikas Sridhar's DGX Spark fleet, 2026-09-05. See `NOTICE`.
