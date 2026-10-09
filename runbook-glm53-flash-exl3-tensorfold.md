# Runbook: GLM-5.3-Flash EXL3 4bpw on 2× DGX Spark via TensorFold (MiaAI-Lab recipe, DFlash2 drafting)

- **Model:** `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @ pinned rev `078455ff…` (164 GiB, 97 files) — EXL3 routed experts (4 bpw), BF16 elsewhere; + `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4e` drafter (~3 GiB).
- **Engine:** [ashhart/TensorFold](https://github.com/ashhart/TensorFold) `tensorfold-glm53:v0.6.0` — GLM-5.3-Flash (`glm5_next`) CUDA family, TP2 across two Sparks over 100G RoCE.
- **Source recipe:** [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) — stock defaults + two overrides below.
- **Campaign date:** 2026-10-06→09 · head **gx10-141d** (.16), worker **edgexpert-9105** (fabric .1) · repo `~/tf-glmflash`, logs `~/tf-glmflash-{prepare,start}.log`, results `edgexpert-04af:~/tf-campaign/glm53-flash-exl3-tf/`.
- **Endpoint:** `http://<head>:8888/v1` (served name `GLM-5.3-Flash-EXL3`)
- **Serving config:** TP2, context 1,048,576 (KV pool 2,347,008 tokens), 76.5 of ~88 GiB per rank, shared-system-prompt reuse on, first start compiles `tensorfold_roce_v1` + `tensorfold_glm_l2pf_v2` CUDA extensions (~5 min, cached in `~/.cache/tensorfold-glm53/<image hash>`).

## Adaptations (deltas from the recipe)

| item | recipe default | ours | why |
|---|---|---|---|
| `SOCKET_IFNAME` | node default route | `enp1s0f0np0` | our fabric NIC; second port DOWN fleet-wide |
| `WORKER` | LAN address | `vikassridhar@10.10.10.1` | head→worker SSH over the RoCE fabric (key trust added) |
| everything else | — | stock | incl. the recipe's own pinned checkpoint rev |

## Measured results (2026-10-09, our standard battery: prompt set v1, temp 0, thinking off)

### Throughput (mimobench C1/C4/C8, 8 categories)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| 1 | **59.4** | 68.6 | 0.307 |
| 4 | **116.4** | 35.9 | 0.695 |
| 8 | **119.9** | 35.6 | 3.14 |

Per-stream C1: format 92.9 · coding 84.3 · json 80.1 · math 78.9 · reasoning 67.9 · prose 51.0 · summary 48.1 · narrative 45.3 · ceiling-count 105.4 (excluded from agg).

### Cold prefill (unique prefix)

| target | actual prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 3,807 | 3.86 | 986 |
| 8K | 15,161 | 7.95 | **1,907** |
| 32K | 60,910 | 32.5 | **1,877** |

### EXL3 (TensorFold) vs NVFP4 (vLLM TP2) — same fleet, same class of 2-Spark pair (vLLM arm ran on cb98+141d)

| metric | vLLM NVFP4+DFlash2 TP2 (09-19) | **TensorFold EXL3 4bpw TP2 (this)** | Δ |
|---|---|---|---|
| C1 prose | 22.7 | **51.0** | **+125%** |
| C1 coding | 32.1 | **84.3** | **+163%** |
| C4 agg | 51.2 | **116.4** | +127% |
| C8 agg | 72.4 | **119.9** | +66% |
| prefill @32K | not measured | 1,877 | — |
| quant fidelity (published KLD vs BF16, GLM-5.3-Flash panel) | NVFP4 0.0605 nats | **EXL3 0.0246 nats** | 2.5× less drift |

**Headline: the EXL3 lane wins on both axes** — 2.2–2.6× the decode speed of our NVFP4 vLLM arm *and* measurably closer to BF16 (per the published 25-window KLD panel; EXL3 4bpw lands within 0.004 nats of official FP8). Caveat: the recipe's own headline claims (prose 121.8 / code 165.3) are far above our C1 — those are best-case per-category peaks with warm prefix reuse; our battery numbers (temp 0, cold unique prefixes) are the honest floor. C8 TTFT inflates (3.1 s): the engine batches fewer streams than vLLM at high concurrency; C4/C8 aggregate gains flatten past 4 streams.

## Deploy steps (deltas from the recipe README)

1. Clone recipe; `scripts/local.sh`: `WORKER=vikassridhar@10.10.10.1`, `SOCKET_IFNAME=enp1s0f0np0`, `WORKER_WEIGHTS=copy`.
2. `./scripts/prepare.sh` — downloads checkpoint+drafter on head (needs a writable HF cache, see appendix), rsyncs ~164 GB to the worker, verifies file-by-file, builds/ships the image on both.
3. `./start.sh` → first start compiles the CUDA extensions (~5 min) → LIVE on :8888.
4. Bench from an idle node: `mimobench.py --base http://10.10.10.16:8888/v1 --model GLM-5.3-Flash-EXL3 --levels 1,4,8 --prefill 2000,8000,32000`.

## Appendix: bugs & fixes

1. **prepare.sh loops forever on `PermissionError: /hf/hub/.locks/...`** — the HF cache had root-owned dirs from an earlier in-container download; the script retries the doomed download indefinitely (no fail-fast). Fix without sudo: `docker run --rm --privileged --pid=host --entrypoint bash <any-local-image> -c 'nsenter -t 1 -m -- chown -R 1000:1000 ~/.cache/huggingface'`, kill the looping prepare, relaunch. Watch for this on any node that ran container-side HF downloads.
2. **`hf download` wedges silently on large multi-file repos** (fleet-wide, seen on 629-file repos): process sleeps, zero bytes, empty log, even with `HF_HUB_DISABLE_XET=1`; single files work. Workaround: enumerate `api/models/<repo>` siblings → parallel `curl -C -` with HEAD-size skip (resumable); or let the recipe's in-container downloader do it (worked here: 97 files / 164 GB in ~1h54m).
3. **Worker disk:** the `copy` mode needs the full 164 GB + image (~25 GB) on the worker — 9105 landed at 104 GB free after; check before choosing copy vs nfs.
4. Host DNS on the control box can flap ("Temporary failure in name resolution") — address fleet nodes by fabric IP (head .16, worker .1) when it does; nothing wrong on the nodes.
