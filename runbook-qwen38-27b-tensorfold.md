# Runbook: Qwen3.8-27B on a single DGX Spark via TensorFold (MiaAI-Lab recipe, DFlash2 drafting)

- **Model:** `Vontra/Qwen3.8-27B-MLX-4bit` (~15 GiB) + `z-lab/Qwen3.8-27B-DFlash2` drafter (3.6 GiB, 81 tensors) — MLX 4-bit weights read by TensorFold's CUDA engine on GB10, DFlash2 speculative drafts.
- **Engine:** [ashhart/TensorFold](https://github.com/ashhart/TensorFold) **0.6.0** — int8/fp8 KV, DFlash2 + context-copy drafts, SSD n-gram lookup.
- **Source recipe:** [MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold) — stock defaults.
- **Campaign date:** 2026-10-06 · node **edgexpert-9105** (repo `~/tf-qwen27b`, log `~/tf-trackB.log`, results `edgexpert-04af:~/tf-campaign/qwen38-27b-tf/`).
- **Endpoint:** `http://<node>:8888/v1` (served name `Qwen3.8-27B`)
- **Serving config (recipe defaults):** 8 streams × 262,144 tokens, fp8 KV, KV pool auto-pinned at 78 GiB, DFlash2 drafting on.

## Measured results (2026-10-06, our standard battery: prompt set v1, temp 0, thinking off)

### Throughput (mimobench C1/C4/C8, 8 categories)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| 1 | **80.1** | 85.4 | **0.133** |
| 4 | **184.4** | 57.2 | 0.495 |
| 8 | **243.9** | 41.4 | 0.983 |

Per-stream C1 by category: format 127.5 · math 115.0 · coding 114.3 · json 111.0 · reasoning 81.3 · summary 49.8 · prose 43.2 · narrative 40.9 · ceiling-count 179.3 (excluded from agg).

### Cold prefill (unique prefix)

| target | actual prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 5,075 | 2.72 | 1,866 |
| 8K | 20,117 | 11.21 | 1,794 |
| 32K | 80,629 | 57.9 | 1,392 |

**Headline: best single-Spark result in fleet history** — a 27B dense model at 244 tok/s aggregate C8 with 0.13 s TTFT, and 115 tok/s C1 on math/coding categories (draft acceptance is content-driven; structured tokens verify best). Narrative/prose sit ~40–43 (drafts accept less on free-form text), matching the pattern seen on every spec-decode stack we run.

## Deploy steps (deltas from the recipe README)

1. Clone recipe → `./start.sh` (prepare downloads model+drafter into the HF cache, builds the image, serves on :8888).
2. Bench from an idle node: `mimobench.py --base http://10.10.10.1:8888/v1 --model Qwen3.8-27B --levels 1,4,8 --prefill 2000,8000,32000`.

## Appendix: bugs & fixes (the drafter-download saga)

1. **HF xet backend failure** — `hf download` of the DFlash2 drafter died with a CAS Client Error / hung silently (0 bytes, process sleeping) even with `HF_HUB_DISABLE_XET=1`. Main model cached fine earlier; only this repo's blob failed. Fix path: bypass the CLI entirely — `curl -L <resolve-url>` straight into the snapshot dir.
2. **curl exit 23 (write error)** — the snapshot dir was **root-owned** (created by an earlier in-container download). Fix without sudo: `docker run --rm --privileged --pid=host --entrypoint bash <any-local-image> -c 'nsenter -t 1 -m -- chown -R 1000:1000 <dir>'`. Use a locally-present image — pulling a 20 GB image just to chown wastes 10 min.
3. **Full-repo `hf download` wedges on 629-file repos** (seen on 141d, appendix of the Flash-Next runbook): single files download fine, full snapshot hangs with empty log. Workaround: enumerate `api/models/<repo>` siblings → parallel `curl -C -` with size-verified skips (resumable).
4. Validate a curl'd safetensors before serving: read the 8-byte header length, parse the JSON header, count tensors — catches truncated downloads.
