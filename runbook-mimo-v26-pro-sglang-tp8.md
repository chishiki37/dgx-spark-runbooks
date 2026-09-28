# Runbook: MiMo-V2.6-Pro-RL on 8× DGX Spark via SGLang (rhys101 a14 recipe, single-rail adaptation)

- **Model:** `XiaomiMiMo/MiMo-V2.6-Pro-RL` rev `73875d00…` (pre-MOPD checkpoint, 535 GB, 155 files) — **full local copy required on every node** (the recipe's fast-expert-load patch reads local; NFS-shared weights are not what it was built for).
- **Source recipe:** [rhys101/MiMo-V2.6-Pro-RL-Spark-8](https://github.com/rhys101/MiMo-V2.6-Pro-RL-Spark-8) build `a14` — patched SGLang (`lmsysorg/sglang` dev-cu13 nightly 2026-09-23, commit `06008c17`, carries MiMo-V2.6 day-0 support sglang#40448) + 12 patches (6 upstream PRs, RoCEnante b12x hook, fast expert load, marlin zero-fill skip, FP8 o_proj/draft, adaptive DFlash verify width, chunked-prefill admission fix) + GB10-tuned Triton FP8 tile tables.
- **Campaign date:** 2026-09-28 · campaign dirs on edgexpert-04af: `~/mimo-v2.6-pro/` (repo + build), `~/mimo-campaign/sglang-a14/` (results), `~/aevonix-ref/` (extraction fixtures).
- **Hardware:** same 8× Spark fleet as the vLLM campaigns — 04af (10.10.10.15, head/rank0/API) · 3b24 (.12) · ae1e (.13) · 9105 (.1) · bdea (.2) · cb98 (.14) · 141d (.16) · 1d49 (.11). **Single 100G RoCE rail** (`rocep1s0f0`/`enp1s0f0np0`, 10.10.10.0/24) vs the recipe's assumed dual 200G subnets.
- **Endpoint:** `http://10.10.10.15:30000/v1` (served name `mimo-v2.6-pro`)

## Adaptations from the upstream recipe (deltas only)

| item | upstream a14 | ours | why |
|---|---|---|---|
| `SGLANG8_ROCE_ALLREDUCE` | 1 (RoCEnante) | **0** | hard requirement: "SG17 RoCEnante requires two active RoCE interfaces" — we have one rail. Plain NCCL all-reduce instead |
| `NCCL_IB_HCA` | `rocep1s0f0:1,roceP2p1s0f0:1` | `rocep1s0f0:1` | single rail |
| NCCL fabric env | theirs | ours from the vLLM TP8 campaign (`GID_INDEX=3`, `ROCE_VERSION_NUM=2`, `ADDR_RANGE=10.10.10.0/24`, NVLS/CROSS_NIC/MERGE off, CUMEM off) | proven on this fleet |
| FABRIC_A/B in cluster.local.env | two subnets | both = our 8 fabric IPs | their cluster.sh interleaves A/B for docker-distribute; identical lists are safe |
| everything else | — | **exact a14 defaults** | TP8/EP1, marlin MoE, FA4 prefill, triton decode attention, DFlash-8 adaptive, FP8 o_proj+draft, 262K ctx, mem 0.85, page 64, vision on |

Everything else ran as `configs/cluster.local.env` overriding only the above; serving profile untouched.

## Measured results (2026-09-28, prompt set v1, temp 0, thinking off)

### Main battery (their `bench/mimobench.py`, C1/C4/C8 + cold prefill)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| 1 | 46.7 | 54.3 | 0.406 |
| 4 | 111.0 | 34.2 | 0.716 |
| 8 | **157.4** | 24.1 | 0.942 |

Per-stream C1: format 94.2 · ceiling-count 108.5 (excluded from agg) · math 73.1 · json 67.8 · coding 55.9 · reasoning 41.5 · summary 37.3 · prose 35.5 · narrative 29.2.
Cold prefill: ~1,160–1,170 tok/s @32K target (TTFT-based; their filler overshoots — actual 81K prompt tokens).

### Long structured extraction (Aevonix fixtures, exact-match vs oracles, cold cache)

| metric | ours (single-rail) | rhys101 a14 published | Δ |
|---|---|---|---|
| C1 e2e tok/s (3 reps) | 83.1, **12/12 correct** | 104.6 | −21% |
| C8 aggregate (3 waves ×2) | **260.1, 24/24 correct** | 306.7 | −15% |

### Quality battery (same harnesses as the DS4 campaigns)

- **GSM8K (50 q, greedy): 48/50 = 96% true** — raw 44/50; of the 6 misses, 4 are string-vs-float scoring artifacts (`26.00` vs `26`, pitfall #36b) and **2 are genuine arithmetic errors** (Q12 got 12 vs 13, Q15 got 96 vs 125). rhys101 reference: 98.0% (200 q) — same ballpark.
- **HumanEval (50 tasks, chat endpoint): 49/50 = 98.0%** — sole failure HumanEval/32, the known shared cross-checkpoint weakness.
- Smoke (`runtime/smoke.py`): arithmetic, reasoning, tool-call JSON well-formed, image (red/blue halves) — all pass.

### Vs our vLLM TP8 arms (same fleet, same battery where comparable)

| metric | vLLM triton (delivered 09-22) | vLLM marlin+EP (09-28) | **SGLang a14 single-rail** |
|---|---|---|---|
| C1 aggregate | 30.2 | 31.4 | **46.7** |
| C4 aggregate | 54.2 | 62.7 | **111.0** |
| C8 aggregate | 71.4 | 78.7 | **157.4** |
| C1 TTFT | 0.54 s | 0.47 s | **0.41 s** |
| C1 coding per-stream | 55.9 | 49.4 | 55.9 |

**SGLang roughly doubles the best vLLM arm at C4/C8 even with the single-rail handicap.** The stack differences that matter: packed MXFP4 experts on Marlin (no FP8 expert expansion), DFlash-8 with per-request adaptive verify width (4 vs 8 tokens based on running acceptance), FA4 prefill, FP8 o_proj/draft weights.

## Deploy steps (deltas from upstream README quick-start)

1. Free the fleet (DS4/H3/vLLM containers down; `cluster.sh serve` refuses on busy GPUs — `nvidia-smi --query-compute-apps` must be empty on all 8).
2. Disk: 535 GB free per node for the model + ~70 GB for the image (containerd-store nodes report larger after `docker load` — cosmetic, content-verify with `applied-prs.txt` + `git rev-parse` inside the image, not the ID).
3. Stage the repo on the head (`~/mimo-v2.6-pro`), write `configs/cluster.local.env` (see adaptations above).
4. **Build locally on the head** — `cluster.sh build` rsyncs to BUILD_NODE and its fabric guard rejects head-as-build-node (route to own fabric IP goes via `lo`). Manual equivalent: `cp -r docker patches runtime third_party build/ && cd build && docker build -f docker/Dockerfile -t mimo26-spark:a14 .` (~50 min).
5. Model fan-out: `STREAMS=4 bash scripts/copy-model.sh` — tree-shaped (head→r1/r2, then r1/r2→r5-7). **Prerequisite: node-to-node passwordless SSH along the tree branches** (r1→r5/r6, r2→r7); a missing key silently fails that branch (`Permission denied` in `logs/copy-rankN.log`) while the script exits 0 — always verify `du -sh` on every node after. Fallback: direct head→node rsync for the missing branches.
6. Distribute image: `docker save mimo26-spark:a14 | ssh <node> docker load`, 7-way parallel over the fabric (~40 min), then verify IDs match (containerd-store exception above).
7. `bash scripts/cluster.sh serve` — workers-first (rank 7→0), waits `/health` up to 60 min. **Boot ≈ 9–14 min** with local weights.
8. `python3 runtime/smoke.py`, then benches. NOTE: `long_extraction_c1/c8.py` take **positional** args `BASE_URL REPO_DIR REPEATS|WAVES` where REPO_DIR is a checkout of [Aevonix/mimo-2.6-dgx-spark](https://github.com/Aevonix/mimo-2.6-dgx-spark) (fixtures + benchmark.py).
9. Teardown: `bash scripts/cluster.sh stop` (removes all `mimo26-r*`, waits for GPU procs to clear).

## Failure ladder

- **S1 — RoCEnante needs 2 rails:** `RuntimeError: SG17 RoCEnante requires two active RoCE interfaces` at scheduler init (fails fast, clean kill, no leak). Fix: `SGLANG8_ROCE_ALLREDUCE=0` on single-rail fleets; expect ~15-21% less throughput than published dual-rail numbers (their profile: all-reduce was 14% of decode step).
- **S2 — `cluster.sh build` guard vs head-as-build-node:** `non-fabric route to <own-fabric-ip> via lo`. Build manually on the head (step 4).
- **S3 — copy-model.sh silent partial failure:** exit 0 with branches skipped when intermediate nodes lack SSH keys; check per-node `du -sh` + `logs/copy-rank*.log` for `Permission denied`.
- **S4 — bench script arg style:** `long_extraction_*.py` are positional; passing `--base/--model` crashes instantly (`invalid literal for int()`).
- **S5 — GSM8K harness scoring:** string compare marks `26.00` wrong vs `26` — recompute true score before quoting (pitfall #36b of the DS4 skill).

## Notes

- **Pre-MOPD checkpoint:** Xiaomi's 2026-09-27 post documents tool-call repetition/flooding in V2.6 RL checkpoints (Pro-RL 0.05-0.54% by harness) fixed by MOPD-suffixed checkpoints on HF. Our rev `73875d00` (and rhys101's, and Aevonix's) are pre-fix. No speed impact; matters only for multi-turn agentic use. Quality battery above shows no single-turn degradation.
- The `--enable-ep-weight-filter`-style knobs from the vLLM world don't exist here; SGLang's EP=1 + packed-MXFP4 marlin is the equivalent path.
- Raw results on 04af: `~/mimo-campaign/sglang-a14/bench-sglang-a14.{json,md}`, `long_extraction_c{1,8}.log`, `~/mimo-campaign/sglang-a14-quality/`.
