# Runbook: GLM-5.3 (753B full) on 4× DGX Spark via gr8 v0.5 (knapcio recipe, EXLR8 `home`)

- **Model:** `kindlingai/glm-5.3-exlr8-k2-k3-k4` @ `a2b599e06c59`, width manifest **`home`** (~3.25 bpw eff., 310 GB total, ~75 GB experts/rank at TP=4). EXLR8 format: QTIP-style trellis codes with FP8 E4M3-rounded codebook; requires an E4M3-rounding loader (gr8, TensorFold, MLXLR8).
- **Engine:** [knapcio/GLM-5.3-gr8-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-gr8-4x-DGX-Spark-TP4) **v0.5** — upstream [kindlingai/gr8](https://github.com/kindlingai/gr8) @ `6448a16` + 26 patches (GEMV splits, INT4 draft heads, attention rounds 1–3, DFlash2 prefix-reuse, paced L2 prefetch, MoE split table, DSel cost model). Drafter: `incoai/GLM-5.3-DFlash2` @ `425aa615ce32` (CC BY-NC-ND — noncommercial only).
- **Campaign date:** 2026-10-17 · nodes **04af (rank0/head/API .15) + bdea (.2) + ae1e (.13) + 1d49 (.11)**, RoCE fabric. Recipe needs **both rails active** (10.10.10.x + 10.10.20.x per box).
- **Endpoint:** `http://10.10.10.15:8100/v1` (served names `glm-5.3` / `glm53`), **max_model_len 310,514**.
- **Serving config:** recipe's stock `G6ENV_X` lever line incl. `G6_S2_DSEL_TABLE` (knapcio's measured ms table — machine-specific, re-measure on other hardware), `SPEC_ACCEPT=rejection`. Host networking; API on 8100 has **no auth** — trusted fabric only.

## Measured results (2026-10-17, our v1 battery: 9 categories, temp 0, streaming, usage-block tokens)

⚠️ GLM-5.3's chat template has **no thinking-off mode**. The gr8 front end rejects `chat_template_kwargs:{enable_thinking:false}`; harness sends **`reasoning_effort:"low"`** instead (same as knapcio's RigMark config). Not a like-for-like mode vs the thinking-off TensorFold baseline — per-stream decode tok/s are real generated tokens either way.

### Headline vs our previous full-GLM-5.3 arms

| metric | vLLM cuda-exl3+DFlash2 (09-07) | TensorFold TP4 EXL3-2.75bpw (10-06) | **gr8 v0.5 EXLR8 home (this)** | knapcio RigMark v0.5 (ref) |
|---|---|---|---|---|
| C1 agg | 16.4 | 37.1 | **47.4** | 46.1 |
| C1 per-stream | — | 40.7 | **55.9** | 51.5 |
| C4 agg | — | n/a (single-stream) | **63.4** (22.3/stream) | 72.5 (22.0/stream) |
| C8 agg | 56.4 @200K | n/a | **74.5** (15.5/stream) | — |
| prefill @2K/8K/32K/64K | — | 1107/1099/1052/— | 703/936/**1145**/1114 | — (1359/1383/1316/1154 @8K/32K/64K/128K RigMark) |
| context | 200K–1M | 140K (1M w/ DCP4, −10% dec) | **310K** | — |

Our C1/C4/C8 numbers track knapcio's RigMark closely (independent harness) — recipe reproduces. Their higher prefill numbers come from RigMark's cold-unique-prompt protocol; our filler prompts land lower at small sizes (2K) but converge by 32K+.

### Per-stream tok/s by category (C1 / C4 / C8)

| category | C1 | C4 | C8 |
|---|---|---|---|
| structured | **78.05** | 23.47 | 18.26 |
| math | 68.43 | 24.43 | 17.00 |
| coding | 67.73 | 24.54 | 16.80 |
| format | 67.85 | 26.89 | 18.16 |
| json | 56.08 | 24.52 | 17.68 |
| summary | 45.64 | 19.21 | 12.88 |
| prose | 42.64 | 21.21 | 13.82 |
| reasoning | 40.49 | 19.53 | 12.89 |
| narrative | 36.60 | 16.86 | 11.76 |
| ceiling_count | 82.96 | 26.94 | 19.13 |

TTFT mean: C1 0.49s · C4 1.93s · C8 4.19s. DFlash2 drafts long blocks on structured/code/math → biggest single-stream wins; narrative/prose gain least (matches recipe docs).

### Cold prefill (unique prefix, filler)

| target | prompt tokens | TTFT (s) | prefill tok/s |
|---|---|---|---|
| 2000 | 1494 | 2.13 | 703 |
| 8000 | 5938 | 6.34 | 936 |
| 32000 | 23844 | 20.83 | 1145 |
| 64000 | 47771 | 42.87 | 1114 |

### Quality

- Recipe-reported (v0.5): qeval 577/600 sampled, 72/75 T=0; 96K needle correct; bit-exact levers byte-verified.
- EXLR8 `home` fidelity (kindlingai card, full-vocab KL vs BF16): held-out mean **0.149**/median 0.0125 — vs EXL3 3.25bpw 0.161 on the same harness; BF16-vs-BF16 batching noise alone scores 0.11 mean. Our old EXL3-2.75bpw arm is materially softer than `home`.
- Our lm_eval attempt (raw `/v1/completions`, n=100) produced low IFEval/MMLU numbers that are **harness artifacts, not model quality**: gr8's completions API has no `echo`, so lm_eval ran untemplated raw continuations — the model leaks its reasoning stream after answers (MMLU letter extraction fails even when the correct letter is chosen first) and instruction-following degrades without the chat template. Verified by probing the live endpoint: same MMLU-Math question answers exactly `"C"` via chat API; IFEval comma-only/phrase constraints pass in chat mode. Results removed from this repo — a chat-mode (`local-chat-completions` + `reasoning_effort:"low"`) rerun is required before publishing any quality numbers.

## Deploy steps (deltas from the recipe README)

1. **Build image on ONE box, distribute via `docker save | ssh docker load`** — parallel buildx on 04af/bdea/ae1e all failed identically (`failed commit on ref ... rename: no such file`) while 1d49 built fine; not worth debugging. 2.83GB image, 1.8GB gz.
2. **No sudo on fleet boxes** → skip `/srv/gr8/*`; leave `GR8_CACHE`/`GR8_HOME` commented in `.env` (Docker named volumes `gr8_cache`/`gr8_home`). Persistent across reboots; `prepare` resumes after power cycles (hash-verifies, fetches only missing bytes).
3. **IPv6 blackhole is the #1 trap**: containers resolve huggingface.co to AAAA first; fleet LAN has no IPv6 egress → HF downloads hang forever (looked dead: 231MB frozen 7.5h, one repo-resolve took 5h). Fix: bind-mount a `gai.conf` with `precedence ::ffff:0:0/96  100` into the container (add volume line to compose.yaml) → IPv4-first, full speed (~3–4 MB/s sustained per box, 85.6GB/rank total incl. retries).
4. **Fabric needs BOTH rails up**: gr8's preflight requires 2 active rail addresses per box (10.10.10.x + 10.10.20.x). After reboot ae1e's `enP2p1s0f0np0` was DOWN → handshake refused cluster-wide. Fix without sudo: privileged alpine + `nsenter -t 1 -m -n ip link set enP2p1s0f0np0 up`. Check all 4 boxes before `up`.
5. `.env` on every box: `PEERS=10.10.10.15,10.10.10.2,10.10.10.13,10.10.10.11`, stock `G6ENV_X`, `SPEC_ACCEPT=rejection`. Identical everywhere (split-flag mismatch hangs boot).
6. `prepare` per box (~90GB+4.9GB drafter each, 2–4h at HF speeds) → `docker compose up -d` on all 4 → engines ready in ~49s after handshake → healthcheck flips to serving (state file `/var/lib/gr8/home/out/gr8.state`).
7. Warning in logs: `vm.compaction_proactiveness is 20` — wants root `sysctl -w vm.compaction_proactiveness=0`. Not applied (no sudo); no stall observed in this campaign.

## Bench commands

```bash
# v1 battery (mimobench.py harness, patched: chat_template_kwargs → reasoning_effort low)
python3 gr8bench.py --base http://10.10.10.15:8100/v1 --model glm-5.3 \
  --label gr8-glm53-tp4-exlr8-home --levels 1,4,8 --prefill 2000,8000,32000,64000 --out ~/gr8-campaign
# smoke
curl -s http://10.10.10.15:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"Reply with exactly: SMOKE_OK"}],"temperature":0,"max_tokens":500}'
```

## Appendix: bugs & fixes (failure ladder)

1. **buildx blob-commit failure on 3 of 4 boxes** — identical error at the CUDA base layer; worked on 1d49. Fix: build once, `docker save`+gzip (1.8GB), scp over fabric, `docker load`. Needs inter-node SSH trust (propagate pubkeys first).
2. **HF downloads hang at ~0 B/s inside container** — IPv6-first resolution, no IPv6 egress. Host `curl -4` = 22–35 MB/s proved the path. Fix: `gai.conf` precedence bind-mount (step 3 above).
3. **Handshake refused: "fabric: 1 active rail address, want two"** on ae1e post-reboot — second rail link down. Fix: nsenter link-up (step 4).
4. **`enable_thinking:false` → HTTP 400** from gr8 front end ("no switch for it: use reasoning_effort") — patch harness to send `reasoning_effort:"low"`.
5. **lm_eval `mmlu_generative_stem` not a task** — use per-subject `mmlu_*_generative` tasks. Also lm-eval-venv missing `transformers` (needed for HF tokenizer backend even for API models): `uv pip install transformers` (5.20.1, tokenizer-only is enough).
6. **gr8 `/v1/completions` has no `echo`** — loglikelihood-based lm_eval tasks impossible; stick to generative tasks. Raw completions work (needed for lm_eval's local-completions model).
7. **lm_eval on THIS host needs an IPv4-first DNS wrapper** — same IPv6 blackhole as the Sparks: dataset/tokenizer fetches from HF hang in SYN-SENT forever. `gai.conf` alone didn't fix Python; use `~/gr8-campaign-results/lmeval_v4.py` (monkeypatches `socket.getaddrinfo`). Also required: `HF_ALLOW_CODE_EVAL=1`, `PYTHONSAFEPATH=1` (nltk defusedxml cwd block), `--confirm_run_unsafe_code` (mbpp gate in this lm_eval version), and `tokenizer=kindlingai/glm-5.3-exlr8-k2-k3-k4` in model_args (bare served name 404s on HF).
