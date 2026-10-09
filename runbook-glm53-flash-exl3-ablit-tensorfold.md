# Runbook: GLM-5.3-Flash EXL3 4bpw **Abliterated** on 2× DGX Spark via TensorFold (MiaAI-Lab recipe, ABLIT=1)

- **Model:** `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit` @ pinned rev `57edefd2…` (gated repo; 175.7 GB, 97 files) + `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4e` drafter.
- **Provenance:** Mia-AiLab's own EXL3 4 bpw quant of `zai-org/GLM-5.3-Flash` (routed experts only — `scope: glm53_routed_experts_only`, BF16 attention) with drowzeys' 30 ablated BF16 `o_proj` tensors grafted in (layers 15–43 + MTP 45). **No quant→quant conversion:** the ablation edit is BF16-space; the NVFP4 in drowzeys' repo name is their original distribution target. drowzeys reports 32/32 Refusal32 bypass on the NVFP4 parents; that score does not transfer as a measured claim to this EXL3 repack (we did not re-run Refusal32).
- **Engine:** [ashhart/TensorFold](https://github.com/ashhart/TensorFold) `tensorfold-glm53:v0.6.0`, TP2 over 100G RoCE.
- **Source recipe:** [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) — the recipe has a **first-class `ABLIT=1` switch** (config.sh pins the ablit rev, defaults THINKING=0 for it).
- **Campaign date:** 2026-10-09 · head **gx10-141d** (.16), worker **edgexpert-9105** (.1) · results `edgexpert-04af:~/tf-campaign/glm53-flash-exl3-ablit-tf/`, this repo `results/tensorfold-glm53-flash-exl3/bench-tf-glm53flash-exl3-ablit.*`.
- **Endpoint:** `http://<head>:8888/v1` (served name `GLM-5.3-Flash-EXL3`), thinking off by default (ABLIT=1 flips the recipe default).

## Deploy steps (deltas from the parent EXL3 runbook)

1. **Gate access:** the ablit repo is gated — accept the Responsible Use form on HF with your account, token needs `read` scope. Probe: `curl -H "Authorization: Bearer <tok>" .../resolve/main/config.json` → 200.
2. **Download** on the head with the parallel-curl downloader (`hf_curl_download.py`, 4 workers, `--token-file`): 100 files / 175.7 GB in 70 min (42 MB/s WAN). Log ends `DOWNLOAD-DONE` + per-file size verify.
3. **Stage into the HF cache layout the recipe expects** (hardlinks, no extra disk): `snapshots/<pinned-rev>/` + `refs/main`. Verify `main` sha == the recipe's pin (`57edefd2…`) via the HF API before trusting the local dir. Hardlink-staging made prepare.sh skip its own 2 h in-container download entirely.
4. **Worker copy:** plain single-stream rsync over the fabric = ~500 MB/s (AES-bound, not link-bound; 175 GB ≈ 10 min). Faster lanes if it matters: NFSoRDMA export/mount (1.2–8.4 GB/s) or 4–8 parallel rsync streams. Byte-verify src==dst after.
5. `./stop.sh` the parent-checkpoint container (it holds ~112 GiB/node), then `ABLIT=1 HF_TOKEN=*** ./start.sh` — LIVE on :8888 in ~5 min (image + kernels cached from the parent campaign; prepare's file-by-file worker verify passes instantly on the pre-staged copy).
6. Smoke: `/v1/models`, then a 17×23 canary with generous `max_tokens` → `391` ✓ (0.2 s).
7. Bench from an idle sibling node (we used 04af over the fabric).

## Measured results (2026-10-09; mimobench prompt set v1, temp 0, thinking off)

### Headline (this run's harness: 9 categories incl. `structured`)

| C | aggregate tok/s | per-stream tok/s | mean TTFT (s) |
|---|---|---|---|
| 1 | 60.4 | 70.6 | 0.32 |
| 4 | **129.9** | 41.5 | 0.53 |
| 8 | 123.5 | 38.3 | 2.91 |

### Speed vs the parent (non-ablit) checkpoint — like-for-like on the parent's 8-category set

The parent campaign ran an older 8-category mimobench; the current script adds `structured` (a fast category, 101.5 tok/s C1), which inflates the ablit headline. Normalizing the ablit run to the parent's exact category set:

| C | ablit (per-stream, 8-cat) | parent (per-stream) | Δ |
|---|---|---|---|
| 1 | 66.8 | 68.6 | −2.6% |
| 4 | 38.3 | 35.9 | +6.7% |
| 8 | 35.5 | 35.6 | −0.3% |

**Ablit ≈ parent speed, within noise** — expected, since the graft changes 30 weight tensors' values, not shapes or kernel paths. The point of this checkpoint is refusal behavior, not throughput. (Category detail: coding C1 77.2 vs 84.3 and reasoning C1 60.2 vs 67.9 are the visible per-category drops; json/math/format are flat-to-better.)

### Cold prefill (unique prefix; current-script token targets — actual token counts differ from the parent table because the older script's filler ratio differed)

| target | actual prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2K | 1,488 | 0.93 | 1,605 |
| 8K | 5,932 | 3.33 | 1,780 |
| 32K | 23,838 | 12.39 | **1,924** |

Parent arm for reference: 986 / 1,907 / 1,877 at 3,807 / 15,161 / 60,910 tokens — same engine, same ballpark; not strictly comparable per row.

### Supplementary (2-prompt prose/code battery, cb98 → head, same boot)

| cell | prose | code |
|---|---|---|
| C1 agg | 51.8 | 88.8 |
| C4 agg | 173.3 | 216.3 |
| C8 agg | 172.6 (TTFT 6.2 s) | 289.4 (TTFT 3.8 s) |

Different harness/prompt set from mimobench — do not cross-compare with the tables above.

## Notes & caveats

- `PARALLEL=4` (recipe default for 2 Sparks); C8 exceeds the stream pool so TTFT inflates (2.9–6.2 s depending on harness) while aggregate stays flat past C4.
- Thinking is OFF by default on this lane (`ABLIT=1` flips `THINKING`); pass `THINKING=1` or per-request to re-enable.
- We did **not** run a refusal benchmark on this deployment; the 32/32 figure quoted in provenance is drowzeys', measured on the NVFP4 parents.
- Same-host caveat: parent and ablit arms both ran on 141d+9105 (head/worker roles identical), so the speed comparison is same-hardware.

## Appendix: bugs & fixes (this campaign only; parent runbook's appendix still applies)

1. **prepare.sh re-downloads 176 GB even when the files exist locally** — unless they sit in the HF-cache snapshot layout at the *pinned* revision. Hardlink-staging (`cp -al`) the curl-downloaded dir into `hub/models--…/snapshots/<pin>/` + writing `refs/main` makes prepare verify-and-skip. Confirm the pin matches Hub `main` first (API `sha` field), else you stage the wrong rev under a right-looking name.
2. **Single-stream rsync over the 200G fabric caps ~500 MB/s** (one-core AES) — plan NFSoRDMA or parallel streams for >100 GB copies.
3. **Redaction-sensitive launchers:** scripts that inline `HF_TOKEN=*** …)` get masked-mangled by some agent write pipelines; keep the token read inside the remote shell (`export HF_TOKEN=*** ~/.hf_token_ablit)`) and verify the staged script with `bash -n` + visual grep before executing.
