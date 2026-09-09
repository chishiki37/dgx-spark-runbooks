# GLM-5.3 Int4-Int8Mix (743B) — TP4+MTP with gx10 NFS-served weights

**Status:** validated 2026-09-09 · **Nodes:** 4 (gx10-141d head + 3 NFSoRDMA workers) · **Decode:** 26.3 C1 / 56.1 C4 agg / 89.2 C8 agg (2-run means) · **Quality:** GSM8K 98 · HumanEval 92 (n=50) · **Context:** 200K (1M in config, KV-pinned pool 200,064 tok)

GLM-5.3 (743B, 78 layers, `GlmMoeDsaForCausalLM`, 256 experts/8 active) QuantTrio-policy Int4-Int8Mix
(compressed-tensors, 405 GB, 282 shards, MTP draft block included — `model.layers.78.*`), served on
4× DGX Spark with **one copy of the weights on gx10-141d, NFSoRDMA-exported to the workers**.

Supersedes the all-local-NVMe lane (27.2/59.1/86.9 on 1d49+3b24+cb98+04af, 2026-08-30): this topology
matches-or-beats it while keeping 3× ~405 GB of NVMe copies off the workers, and its winner config
(dual-rail NCCL + Marlin atomic-add + seqs16) is new — measured here for the first time.

## Topology

```
gx10-141d (10.10.10.16, 200G) — rank 0 head + API :8000 + NFS server
  weights LOCAL ~/models/GLM-5.3-Int4-Int8Mix, bind-mounted at /mnt/glm53-int4mix
  ├── NFSoRDMA (proto=rdma, port 20049, vers=3) ──→ bdea  (10.10.10.2,  200G) rank 1
  │                                                  04af  (10.10.10.15, 100G) rank 2
  │                                                  ae1e  (10.10.10.13, 100G) rank 3
  └── all workers mount /mnt/glm53-int4mix (same path as head's bind-mount → uniform launcher)
```

- Head reads weights locally (the NFS server never mounts its own export — loopback self-mounts wedge Docker).
- Bind-mount on head + NFS mount on workers at the **same path** keeps one launcher identical on all ranks.
- Head choice: highest-bandwidth IB node (200G) — measured +51% aggregate vs a 100G head on prior campaigns.

## NFS server (gx10)

```bash
sudo apt install -y nfs-kernel-server
sudo sed -i 's/^RPCNFSDCOUNT=.*/RPCNFSDCOUNT=64/' /etc/default/nfs-kernel-server   # may not apply — see pitfalls
echo '/home/vikassridhar/models/GLM-5.3-Int4-Int8Mix 10.10.10.0/24(ro,sync,no_subtree_check,no_root_squash) 10.10.20.0/24(ro,sync,no_subtree_check,no_root_squash)' | sudo tee -a /etc/exports
sudo systemctl enable --now nfs-kernel-server && sudo systemctl restart nfs-kernel-server
# RDMA enable AFTER the final restart (restart wipes the portlist):
sudo bash -c 'modprobe svcrdma; echo "rdma 20049" > /proc/fs/nfsd/portlist'
sudo bash -c 'echo 64 > /proc/fs/nfsd/threads'    # runtime, if /etc/default didn't take
# uniform path for the head rank (bind, NOT self-NFS):
sudo mkdir -p /mnt/glm53-int4mix
sudo mount --bind /home/vikassridhar/models/GLM-5.3-Int4-Int8Mix /mnt/glm53-int4mix
sudo sysctl -w vm.vfs_cache_pressure=200
```

Persist across reboots: `/etc/modules-load.d/nfs-rdma.conf` (`svcrdma`) + a oneshot `nfs-rdma-port.service`
(After=nfs-server) re-adding `rdma 20049`, + fstab bind line.

## NFS clients (each worker)

```bash
sudo tee /etc/sysctl.d/99-nfs-rdma.conf <<< 'sunrpc.tcp_slot_table_entries = 256
sunrpc.tcp_max_slot_table_entries = 256'
sudo sysctl -p /etc/sysctl.d/99-nfs-rdma.conf
sudo sysctl -w vm.vfs_cache_pressure=200
sudo mkdir -p /mnt/glm53-int4mix
sudo mount -t nfs -o ro,vers=3,proto=rdma,port=20049,mountproto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 \
  10.10.10.16:/home/vikassridhar/models/GLM-5.3-Int4-Int8Mix /mnt/glm53-int4mix
nfsstat -m /mnt/glm53-int4mix | grep proto   # MUST show proto=rdma (vers=4 silently falls back to tcp)
```

Measured from bdea: single-stream O_DIRECT 507 MB/s, 4-stream aggregate **3.6 GB/s**. Weight load phase of
boot ≈ 7 min for 3 concurrent NFS ranks + 1 local rank (vs ~5 min all-local — a ~2 min tax for 1.2 TB of
NVMe saved).

## Serve config (WINNER — after 10-cell autoresearch sweep)

Image `vllm-node-tf5-glm52-b12x:probe-modded` (b0deb728372c) + 10 sm12x Triton overlays (`~/glm-triton`,
matched-pair checked) + nccl-2.30.4 `LD_PRELOAD` staged at `/var/tmp/glm53-cache/hub/nccl-2.30.4/`.

```
--tensor-parallel-size 4 --nnodes 4 --master-port 29552   (workers: --headless)
--max-model-len 200000 --max-num-seqs 16 --max-num-batched-tokens 8192
--gpu-memory-utilization 0.91 --kv-cache-memory-bytes 10950000000 --kv-cache-dtype fp8_ds_mla
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_tensor_parallel_size":1,"attention_backend":"FLASHMLA_SPARSE"}'
--compilation-config '{"cudagraph_mode":"FULL"}' --enable-prefix-caching --async-scheduling
--reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --trust-remote-code
```

Env deltas vs the local-weights lane (the three winning knobs):

```
NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0            # DUAL RAIL (was single) — NCCL_IB_GID_INDEX still OMITTED
NCCL_SOCKET_IFNAME=enp1s0f0np0,enP2p1s0f0np0
VLLM_MARLIN_USE_ATOMIC_ADD=1                    # Marlin MoE kernel hint
# max-num-seqs 16 (was 6/8)
```

Unchanged doctrine: GID index omitted (fleet drift), NCCL_MAX/MIN_NCHANNELS=4, GLM52_* Triton env set,
`HF_HUB_OFFLINE=1`, pinned KV (never unpinned — deep-wedge risk), desktops killed + swappiness=10 +
one-time drop_caches pre-launch, **threshold-flushers only** (drop iff MemAvailable<8 GiB; periodic
unconditional flushers livelock NFS mmap loads).

Boot: ~17–20 min (NFS load ~7 min, compile ~70–80 s, FULL graphs ~10 s, engine init ~130–145 s).
KV pool 200,064 tokens — byte-identical to the local lane (same pin, same geometry).

## Sweep results (single-knob arms vs baseline; C1 warm / C4 agg / C8 agg tok/s)

| Arm | C1 | C4 | C8 | Σ | Verdict |
|---|---|---|---|---|---|
| A0 baseline (k3, seqs8, single-rail) | 24.77 | 49.12 | 86.60 | 160.5 | reference |
| A1 seqs16 | 25.15 | 52.51 | 87.52 | 165.2 | stacks |
| A2 MTP k4 | 23.40 | 48.39 | 79.95 | 151.7 | reject — k3 confirmed |
| A3 NCCL nch8 | 25.79 | 50.96 | 80.95 | 157.7 | reject — C8 −6.5% |
| A4 marlin-atomic | 26.58 | 59.59 | 83.82 | 170.0 | C4 king (+21%) |
| A5 enforce-eager | 23.79 | 57.00 | 80.87 | 161.7 | reject — graphs matter |
| A6 dual-rail | 27.66 | 54.27 | 89.88 | 171.8 | best single knob |
| A7 ctx160k | 24.47 | — | — | — | reject — engine died at first concurrent request (chatcmpl KeyError) |
| **COMBO dual+marlin+seqs16 run1** | **27.64** | **55.57** | **92.89** | **176.1** | **winner** |
| COMBO run2 (verify) | 24.94 | 56.66 | 85.47 | 167.1 | C4/C8 repeat; C1 noisy |
| COMBO-B dual+marlin (seqs8) | 25.91 | 53.79 | 79.42 | 159.1 | reject — seqs16 required for C8 |

**Winner = COMBO, 2-run means 26.3 / 56.1 / 89.2 (Σ 171.6).** vs baseline: C1 +6%, C4 +14%, C8 +3%;
best runs: 27.6 / 56.7 / 92.9. vs the all-local-weights lane: C1 −3%/+1.6% (run-dependent), C4 −5%,
C8 +3–7% — NFS parity at C1/C8 with 1.2 TB of worker NVMe freed.

**Variance warning:** single-shot batteries swing ±10% on C1/C8 at identical config (24.94–27.64 C1).
Marlin's C4 edge was the only consistently reproducible gain (3/3 runs ≥55.6). Run ≥2 batteries per arm
before believing C1/C8 deltas under ~10%.

## Quality (winner config, measured 2026-09-09)

Integrity probes (quant + NFS-lane health):

- **Corruption probe: PASS** — ~6K-char generations × 3 passes each in ko/en/tr, **0 bad tokens** across ~54K chars.
- **Long-context needle: PASS** — 34,996-token prompt, all 3 keys (begin/middle/end) retrieved verbatim; prefill ≈550 tok/s, 63.7 s e2e.

Benchmarks (gx10:8000, single stream):

- **GSM8K 98%** (49/50, canonical set — 1 arithmetic slip)
- **HumanEval-chat 92% pass@1** (46/50)

MCQ sanity battery (n=10/task, temp 0) vs hosted **GLM-5.2** on api.z.ai — a cross-generation
reference point, not the same model:

| Task | Local i4mix | Hosted GLM-5.2 |
|---|---|---|
| GSM8K | 90% | 80% |
| HumanEval | 100% | 100% |
| MBPP | 80% | 80% |
| IFEval | 70% | 90% |
| MMLU-STEM | 100% | 100% |
| ARC-Challenge | 100% | 100% |
| HellaSwag | 100% | 100% |

Ties-or-better on 6/7; the IFEval deficit is 2 questions at n=10 — noise-level, not a quant signal.
Decode during the battery: 22–29 tok/s single-stream mixed content, consistent with the 26.3 C1 bench mean.
Raw JSONs: `~/glm53-i4mix-nfs/results/quality/`.

## Pitfalls (new from this campaign)

1. **ctx 160K + FULL cudagraph kills the engine at the first concurrent chat request** —
   `KeyError: 'chatcmpl-*'` in `req_id_to_index` (previously documented only for the NVFP4-KV overlay
   lane; it also triggers at 160K ctx on fp8_ds_mla). 200K ctx with the 10.95 GB pin is the stable value.
2. **Sudo-hub self-truncation race:** `ssh hub "cat pwfile" | ssh hub "cat > pwfile"` (hub itself in the
   propagation loop) truncates the source before the reader drains it — fleet-wide password loss,
   recoverable only from an already-seeded node. Never include the hub in its own propagation list;
   single writer per pipeline.
3. **`RPCNFSDCOUNT` in `/etc/default/nfs-kernel-server` may not apply** when the file was created before
   the package install (ucf conffile dance). Verify `cat /proc/fs/nfsd/threads`; runtime fix
   `echo 64 > /proc/fs/nfsd/threads`.
4. **gx10 runs an HF upload concurrently with zero measurable interference** — WiFi egress (~12 MB/s,
   network-bound) does not contend with RoCE fabric NFS/NCCL; same-NVMe reads <1% of device bandwidth.

## Files

Launcher (env-knobbed, matched-pair kernel preflight, uniform `/mnt/glm53-int4mix`): controller
`~/glm53-i4mix-nfs/launch_glm53_i4mix_nfs.sh` + sweep harness + combo driver + raw cell JSONs
(`~/glm53-i4mix-nfs/results/`). Model weights: `vikasclawd/GLM-5.3-Int4-Int8Mix` on HF.
