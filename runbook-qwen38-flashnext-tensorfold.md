# Runbook: Qwen3.8-Flash-Next on a single DGX Spark via TensorFold (MiaAI-Lab recipe)

- **Model:** `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` (~106 GiB, fetched by the recipe's prepare step) — MLX 4-bit weights with MTP head, read by TensorFold's CUDA engine on GB10.
- **Engine:** [ashhart/TensorFold](https://github.com/ashhart/TensorFold) **0.6.1** — bespoke per-family kernels, OpenAI-compatible server, int8 KV cache, MTP speculative drafts + SSD n-gram (PLE) lookup.
- **Source recipe:** [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) — ran with stock defaults.
- **Campaign date:** 2026-10-05/06 · node **gx10-141d** (repo `~/tf-flashnext`, log `~/tf-trackA.log`, results `edgexpert-04af:~/tf-campaign/qwen38-flashnext-a14/`).
- **Endpoint:** `http://<node>:8888/v1` (served name `Qwen3.8-Flash-Next`)
- **Serving config (recipe defaults):** `--parallel 5 --context 262144 --kv-dtype int8 --mtp-drafts 6 --mtp-confidence 0.60 --ple-on-ssd --vision --thinking`

## Measured results (2026-10-05, our standard battery: prompt set v1, temp 0, thinking off)

### Throughput (mimobench C1/C4/C8, 8 categories)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| 1 | **63.4** | 70.0 | 0.258 |
| 4 | **129.8** | 37.7 | 0.451 |
| 8 | **152.4** | 41.2 | 2.437 |

Per-stream C1 by category: format 95.0 · math 85.4 · coding 84.8 · json 83.8 · reasoning 73.7 · narrative 48.0 · prose 46.9 · summary 42.3 · ceiling-count 115.0 (excluded from agg).

### Cold prefill (unique prefix)

| target | actual prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 5,075 | 2.99 | 1,699 |
| 8K | 20,117 | 9.24 | **2,177** |
| 32K | 80,629 | 40.6 | 1,984 |

### vs published claims and our vLLM arm

| metric | MiaAI claims | ours | our vLLM NVFP4 TP2 (**2 Sparks**) |
|---|---|---|---|
| C1 | 63.6 prose / 96.9 code | 63.4 agg / 84.8 coding | prose 48.1 / code 63.8 |
| C4 | 166 | 129.8 | — |
| prefill | ~2,400 | 1,984–2,177 | ~1,160 (MiMo ref) |

**Headline: one Spark running TensorFold matches or beats our two-Spark vLLM TP2 NVFP4 deployment** (~+32% prose C1, +33% code C1) and comes within 10–20% of the recipe's self-reported prefill. C4 lands 22% under their 166 claim — their number is likely best-category or warm-cache; treat published figures as upper bounds.

## Deploy steps (deltas from the recipe README)

1. `git clone` the recipe; `./scripts/prepare.sh` (pulls image + ~106 GiB checkpoint; the host needs an `hf` CLI or the script downloads inside the container — 141d had neither, see appendix).
2. `./start.sh` → waits for `LIVE`, serves on :8888.
3. Bench from an idle node: `mimobench.py --base http://10.10.10.16:8888/v1 --model Qwen3.8-Flash-Next --levels 1,4,8 --prefill 2000,8000,32000`.

## Appendix: bugs & fixes

1. **No `hf` CLI on the node** — recipe prepare step expects it on the host or falls back to in-container download. Fix: `python3 -m venv ~/.hf-cli/venv && pip install -U huggingface_hub` (hub 2.x ships the `hf` entry point; the `[cli]` extra no longer exists).
2. **`pkill -f <pattern>` self-match footgun** (fleet-wide): a remote `ssh "pkill -f start.sh"` kills the wrapper shell running it (SIGTERM, exit −15/255). Fix: kill by PID, or exclude `bash -c` wrappers; prefer script-file + `bash script.sh` over inline one-liners for anything with `&`/pkill.
3. C8 TTFT inflates (2.4 s) because `--parallel 5` admits 5 streams; C8 queues 3. Raise `--parallel` only if KV pool allows (262K × 5 streams already pins most of it).
